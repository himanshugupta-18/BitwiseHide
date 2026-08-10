"""
Evaluation framework unit tests (Phase 2.7).

Covers:
- metric validity: identical images (MSE 0 / PSNR inf / SSIM 1), known small
  images with hand-computable values, shape mismatches, invalid input
- determinism of every metric and of the comparison service
- Phase 2.4 basic LSB and Phase 2.6 adaptive LSB exact payload recovery
- capacity reporting, capacity-boundary behavior (fits=True at capacity,
  fits=False past it), graceful failure on oversized payloads
- runtime reporting
- a measurement-driven comparison showing the adaptive method concentrates
  changes away from smooth regions (asserted on measured values only — the
  framework itself makes no superiority claims)
- end-to-end Phase 2.3 encrypt -> evaluate -> extract -> decrypt round trips

No external services are involved; all images are generated in memory.
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image

from app.core import adaptive_embedding
from app.core import steganography as phase24
from app.core.evaluation import (
    image_quality,
    image_to_rgb_array,
    mean_squared_error,
    peak_signal_to_noise_ratio,
    structural_similarity,
)
from app.core.exceptions import EncryptionError, EvaluationError
from app.schemas.encryption import EncryptedPayload
from app.services.adaptive_steganography_service import AdaptiveSteganographyService
from app.services.encryption_service import EncryptionService
from app.services.evaluation_service import ComparisonResult, EvaluationService
from app.services.steganography_service import SteganographyService

PASSWORD = "correct-horse-battery-staple"  # noqa: S105 — test fixture, not a real secret
WRONG_PASSWORD = "definitely-not-the-password"  # noqa: S105

# All 256 possible byte values — proves binary fidelity, not just text.
_ALL_BYTES = bytes(range(256))

#: Flat gray used for "smooth" image regions.
_SMOOTH_GRAY = 128


def _make_png(size: tuple[int, int] = (64, 64)) -> bytes:
    """Create a solid-color (smooth) RGB PNG in memory."""
    image = Image.new("RGB", size, color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _half_smooth_half_textured(size: tuple[int, int]) -> Image.Image:
    """Left half is flat gray (smooth), right half is a 0/255 checkerboard."""
    width, height = size
    smooth = Image.new("L", (width // 2, height), _SMOOTH_GRAY)
    checker = Image.new("L", (width - width // 2, height))
    checker.putdata(
        [255 if (x + y) % 2 == 0 else 0 for y in range(height) for x in range(width - width // 2)]
    )
    composite = Image.new("L", size)
    composite.paste(smooth, (0, 0))
    composite.paste(checker, (width // 2, 0))
    return composite.convert("RGB")


def _image_from_pixels(rows: list[list[tuple[int, int, int]]]) -> Image.Image:
    """Build a small RGB PIL image from a literal row of (R, G, B) pixels."""
    width = len(rows[0])
    height = len(rows)
    image = Image.new("RGB", (width, height))
    image.putdata([pixel for row in rows for pixel in row])
    return image


def _rgb_array(png: bytes) -> np.ndarray:
    """Decode PNG bytes into a uint8 RGB array of shape (H, W, 3)."""
    return image_to_rgb_array(Image.open(io.BytesIO(png)).convert("RGB"))


def _region_mse(cover: bytes, stego: bytes, *, region: Callable[[np.ndarray], np.ndarray]) -> float:
    """MSE between cover and stego over a selected pixel region."""
    cover_region = region(_rgb_array(cover))
    stego_region = region(_rgb_array(stego))
    diff = cover_region.astype(np.float64) - stego_region.astype(np.float64)
    return float(np.mean(diff * diff))


# --- Fixtures -----------------------------------------------------------------


@pytest.fixture
def evaluation_service() -> EvaluationService:
    return EvaluationService()


@pytest.fixture
def smooth_png() -> bytes:
    return _make_png()


# --- Metrics ------------------------------------------------------------------


class TestMetrics:
    def test_identical_images_are_lossless(self) -> None:
        image = _image_from_pixels([[pixel, pixel] for pixel in [(10, 20, 30), (200, 150, 5)]])
        metrics = image_quality(image, image)
        assert metrics.mse == 0.0
        assert metrics.psnr == math.inf
        assert metrics.ssim == pytest.approx(1.0)

    def test_psnr_infinite_when_mse_zero(self) -> None:
        cover = _image_from_pixels([[(0, 0, 0)]])
        stego = _image_from_pixels([[(0, 0, 0)]])
        psnr = peak_signal_to_noise_ratio(image_to_rgb_array(cover), image_to_rgb_array(stego))
        assert psnr == math.inf

    def test_known_small_image_mse(self) -> None:
        # One pixel differs by +1 in the red channel only: MSE = 1 / 3.
        cover = _image_from_pixels([[(0, 0, 0)]])
        stego = _image_from_pixels([[(1, 0, 0)]])
        assert image_quality(cover, stego).mse == pytest.approx(1.0 / 3.0)

    def test_known_small_image_psnr_derived_from_mse(self) -> None:
        # PSNR = 10 * log10(MAX^2 / MSE) with MAX = 255 and MSE = 1/3.
        cover = _image_from_pixels([[(0, 0, 0)]])
        stego = _image_from_pixels([[(1, 0, 0)]])
        metrics = image_quality(cover, stego)
        expected = 10.0 * math.log10((255.0**2) / (1.0 / 3.0))
        assert metrics.psnr == pytest.approx(expected, rel=1e-6)
        assert metrics.psnr == pytest.approx(52.9, abs=0.05)

    def test_psnr_reuses_explicit_mse(self) -> None:
        cover = _image_from_pixels([[(0, 0, 0)]])
        stego = _image_from_pixels([[(1, 0, 0)]])
        mse = mean_squared_error(image_to_rgb_array(cover), image_to_rgb_array(stego))
        direct = peak_signal_to_noise_ratio(image_to_rgb_array(cover), image_to_rgb_array(stego))
        reused = peak_signal_to_noise_ratio(
            image_to_rgb_array(cover), image_to_rgb_array(stego), mse=mse
        )
        assert reused == direct

    def test_ssim_identical_is_one(self) -> None:
        image = _image_from_pixels([[pixel, pixel] for pixel in [(10, 20, 30), (200, 150, 5)]])
        assert structural_similarity(image_to_rgb_array(image), image_to_rgb_array(image)) == 1.0

    def test_ssim_black_vs_white_is_near_zero(self) -> None:
        black = Image.new("L", (32, 32), color=0)
        white = Image.new("L", (32, 32), color=255)
        ssim = structural_similarity(image_to_rgb_array(black), image_to_rgb_array(white))
        assert 0.0 < ssim < 0.01

    def test_ssim_monotonic_in_distortion(self) -> None:
        cover = _image_from_pixels(
            [
                [(0, 0, 0), (40, 40, 40)],
                [(80, 80, 80), (120, 120, 120)],
                [(160, 160, 160), (200, 200, 200)],
            ]
        )
        subtle = _image_from_pixels(
            [
                [(0, 0, 0), (41, 40, 40)],
                [(80, 80, 80), (120, 120, 120)],
                [(160, 160, 160), (200, 200, 200)],
            ]
        )
        gross = _image_from_pixels(
            [
                [(0, 0, 0), (200, 0, 0)],
                [(80, 80, 80), (120, 120, 120)],
                [(160, 160, 160), (200, 200, 200)],
            ]
        )
        subtle_ssim = structural_similarity(image_to_rgb_array(cover), image_to_rgb_array(subtle))
        gross_ssim = structural_similarity(image_to_rgb_array(cover), image_to_rgb_array(gross))
        assert subtle_ssim > gross_ssim

    def test_metric_validity_bounds(self) -> None:
        rng = np.random.default_rng(7)
        cover = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        stego = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        mse = mean_squared_error(cover, stego)
        psnr = peak_signal_to_noise_ratio(cover, stego, mse=mse)
        ssim = structural_similarity(cover, stego)
        assert 0.0 <= mse <= 255.0**2
        assert psnr >= 0.0 or psnr == math.inf
        assert -1.0 <= ssim <= 1.0

    def test_shape_mismatch_raises(self) -> None:
        cover = np.zeros((4, 4, 3), dtype=np.uint8)
        stego = np.zeros((4, 5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="identical shapes"):
            mean_squared_error(cover, stego)
        with pytest.raises(ValueError, match="identical shapes"):
            structural_similarity(cover, stego)

    def test_invalid_metric_argument_raises(self) -> None:
        cover = np.zeros((4, 4, 3), dtype=np.uint8)
        stego = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="data_range"):
            structural_similarity(cover, stego, data_range=0.0)
        with pytest.raises(ValueError, match="window_size"):
            structural_similarity(cover, stego, window_size=8)
        with pytest.raises(ValueError, match="max_value"):
            peak_signal_to_noise_ratio(cover, stego, max_value=0.0)

    def test_invalid_image_input_raises(self) -> None:
        with pytest.raises(EvaluationError, match="PIL image"):
            image_to_rgb_array("not an image")  # type: ignore[arg-type]
        with pytest.raises(EvaluationError, match="positive width and height"):
            image_to_rgb_array(Image.new("RGB", (0, 8)))

    def test_metrics_are_deterministic(self) -> None:
        cover = _image_from_pixels([[(x, y, 3) for x in range(8)] for y in range(8)])
        stego = _image_from_pixels([[(x, y + 1, 3) for x in range(8)] for y in range(8)])
        assert image_quality(cover, stego) == image_quality(cover, stego)


# --- Evaluation service -------------------------------------------------------


class TestRoundTrip:
    def test_basic_lsb_exact_recovery(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        result = evaluation_service.compare_methods(cover=smooth_png, payload=b"basic-eval-payload")
        assert result.basic.fits
        assert result.basic.extracted_correctly
        assert result.basic.stego_png is not None

    def test_adaptive_lsb_exact_recovery(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        result = evaluation_service.compare_methods(
            cover=smooth_png, payload=b"adaptive-eval-payload"
        )
        assert result.adaptive.fits
        assert result.adaptive.extracted_correctly
        assert result.adaptive.stego_png is not None

    def test_all_byte_values_recover(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        result = evaluation_service.compare_methods(cover=smooth_png, payload=_ALL_BYTES)
        assert result.basic.extracted_correctly
        assert result.adaptive.extracted_correctly

    def test_empty_payload_recovers(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        result = evaluation_service.compare_methods(cover=smooth_png, payload=b"")
        assert result.basic.extracted_correctly
        assert result.adaptive.extracted_correctly


class TestCapacity:
    def test_capacity_reported_from_phase_constants(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        image = Image.open(io.BytesIO(smooth_png))
        expected_basic = (image.size[0] * image.size[1] * 3) // 8 - phase24.HEADER_SIZE
        expected_adaptive = (
            image.size[0] * image.size[1] * 3 - adaptive_embedding.HEADER_BITS
        ) // 8
        result = evaluation_service.compare_methods(cover=smooth_png, payload=b"x")
        assert result.basic.capacity_bytes == expected_basic
        assert result.adaptive.capacity_bytes == expected_adaptive
        assert result.payload_size == 1

    def test_payload_at_capacity_fits(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        image = Image.open(io.BytesIO(smooth_png))
        capacity = (image.size[0] * image.size[1] * 3 - adaptive_embedding.HEADER_BITS) // 8
        result = evaluation_service.compare_methods(cover=smooth_png, payload=b"m" * capacity)
        assert result.basic.fits
        assert result.adaptive.fits
        assert result.basic.extracted_correctly
        assert result.adaptive.extracted_correctly

    def test_payload_over_adaptive_capacity_is_recorded(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        # Basic LSB's header is 12 bytes vs adaptive's 14, so basic carries
        # two more payload bytes at full capacity.
        image = Image.open(io.BytesIO(smooth_png))
        adaptive_capacity = (
            image.size[0] * image.size[1] * 3 - adaptive_embedding.HEADER_BITS
        ) // 8
        result = evaluation_service.compare_methods(
            cover=smooth_png, payload=b"m" * (adaptive_capacity + 1)
        )
        assert result.basic.fits
        assert result.basic.extracted_correctly
        assert result.adaptive.fits is False
        assert result.adaptive.quality is None
        assert result.adaptive.stego_png is None

    def test_oversized_payload_both_methods_fail(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        result = evaluation_service.compare_methods(cover=smooth_png, payload=b"m" * 100_000)
        assert result.basic.fits is False
        assert result.adaptive.fits is False
        assert result.basic.quality is None
        assert result.adaptive.quality is None

    def test_tiny_image_has_no_capacity(self, evaluation_service: EvaluationService) -> None:
        result = evaluation_service.compare_methods(cover=_make_png((1, 1)), payload=b"x")
        assert result.basic.fits is False
        assert result.adaptive.fits is False
        assert result.basic.quality is None
        assert result.adaptive.quality is None


class TestInvalidInput:
    def test_empty_image_raises(self, evaluation_service: EvaluationService) -> None:
        with pytest.raises(EvaluationError, match="empty"):
            evaluation_service.compare_methods(cover=b"", payload=b"data")

    def test_non_image_bytes_raises(self, evaluation_service: EvaluationService) -> None:
        with pytest.raises(EvaluationError, match="Invalid image data"):
            evaluation_service.compare_methods(cover=b"this is not an image", payload=b"data")

    def test_non_png_raises(self, evaluation_service: EvaluationService) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buffer, format="JPEG")
        with pytest.raises(EvaluationError, match="Only PNG"):
            evaluation_service.compare_methods(cover=buffer.getvalue(), payload=b"data")

    def test_truncated_png_raises(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        result = evaluation_service.compare_methods(cover=smooth_png, payload=b"hidden")
        assert result.basic.stego_png is not None
        with pytest.raises(EvaluationError, match="Invalid image data"):
            evaluation_service.compare_methods(cover=result.basic.stego_png[:16], payload=b"data")


class TestDeterminism:
    def test_comparison_is_deterministic(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        payload = b"determinism-check"
        first = evaluation_service.compare_methods(cover=smooth_png, payload=payload)
        second = evaluation_service.compare_methods(cover=smooth_png, payload=payload)
        for field in (
            "capacity_bytes",
            "payload_size",
            "fits",
            "extracted_correctly",
            "quality",
            "stego_png",
        ):
            assert getattr(first.basic, field) == getattr(second.basic, field)
            assert getattr(first.adaptive, field) == getattr(second.adaptive, field)

    def test_compare_result_shape_is_stable(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        result = evaluation_service.compare_methods(cover=smooth_png, payload=b"shape")
        assert isinstance(result, ComparisonResult)
        assert result.cover_size == (64, 64)
        assert result.basic.method == "basic_lsb"
        assert result.adaptive.method == "adaptive_lsb"

    def test_runtime_reported(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        result = evaluation_service.compare_methods(cover=smooth_png, payload=b"runtime")
        for evaluation in (result.basic, result.adaptive):
            assert math.isfinite(evaluation.embed_seconds)
            assert evaluation.embed_seconds >= 0.0
            assert math.isfinite(evaluation.extract_seconds)
            assert evaluation.extract_seconds >= 0.0


class TestPlacementMeasurement:
    def test_adaptive_concentrates_changes_away_from_smooth_regions(
        self, evaluation_service: EvaluationService
    ) -> None:
        width, height = 64, 64
        cover = _png_bytes(_half_smooth_half_textured((width, height)))
        smooth_width = width // 2
        result = evaluation_service.compare_methods(cover=cover, payload=b"Z" * 300)
        assert result.basic.stego_png is not None
        assert result.adaptive.stego_png is not None

        def smooth_region(array: np.ndarray) -> np.ndarray:
            return array[:, :smooth_width, :]

        basic_smooth_mse = _region_mse(cover, result.basic.stego_png, region=smooth_region)
        adaptive_smooth_mse = _region_mse(cover, result.adaptive.stego_png, region=smooth_region)
        # Basic writes its payload bits row-major, i.e. into the smooth left half
        # first; adaptive ranks the textured right half first. Measured values
        # only — the framework makes no claim, the numbers do.
        assert basic_smooth_mse > 0.0
        assert adaptive_smooth_mse < basic_smooth_mse


# --- Phase 2.3 integration ----------------------------------------------------


class TestPhase23Integration:
    def test_encrypted_round_trip_basic(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        plaintext = "eval-integration-secret"
        payload = EncryptionService().encrypt(plaintext=plaintext, password=PASSWORD).to_bytes()
        result = evaluation_service.compare_methods(cover=smooth_png, payload=payload)
        assert result.basic.fits
        assert result.basic.extracted_correctly
        assert result.basic.stego_png is not None
        extracted = SteganographyService().extract(result.basic.stego_png)
        restored = EncryptedPayload.from_bytes(extracted)
        assert EncryptionService().decrypt(restored, PASSWORD) == plaintext

    def test_encrypted_round_trip_adaptive(
        self, evaluation_service: EvaluationService, smooth_png: bytes
    ) -> None:
        plaintext = "eval-integration-secret"
        payload = EncryptionService().encrypt(plaintext=plaintext, password=PASSWORD).to_bytes()
        result = evaluation_service.compare_methods(cover=smooth_png, payload=payload)
        assert result.adaptive.fits
        assert result.adaptive.extracted_correctly
        assert result.adaptive.stego_png is not None
        extracted = AdaptiveSteganographyService().extract(result.adaptive.stego_png)
        restored = EncryptedPayload.from_bytes(extracted)
        assert EncryptionService().decrypt(restored, PASSWORD) == plaintext
        with pytest.raises(EncryptionError):
            EncryptionService().decrypt(restored, WRONG_PASSWORD)
