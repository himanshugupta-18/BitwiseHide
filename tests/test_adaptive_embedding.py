"""
Adaptive embedding + service unit tests (Phase 2.6).

Covers:
- exact byte round trips (small, large, binary, empty) through PNG images
- determinism: same image + payload -> identical stego; extraction is stable
- adaptive behavior: bits land in the highest-complexity region first, and the
  implementation actually uses Phase 2.5's analyze()
- capacity validation — never silently truncates a payload
- header/metadata integrity: bad magic, unsupported version/flags, invalid
  weights, and impossible lengths all fail closed
- malformed/truncated/non-PNG input fails closed
- format separation: a Phase 2.6 payload is invisible to Phase 2.4 extraction
  and vice versa
- an end-to-end Phase 2.3 encrypt -> adaptive embed -> extract -> decrypt
  round trip, including wrong-password rejection

No external services are involved; all images are generated in memory.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.core import adaptive_embedding
from app.core import steganography as phase24
from app.core.adaptive_embedding import (
    ADAPTIVE_MAGIC,
    FORMAT_VERSION,
    HEADER_BITS,
    HEADER_SIZE,
    embed_bytes,
    extract_bytes,
    max_payload_bytes,
)
from app.core.exceptions import (
    AdaptiveEmbeddingError,
    EncryptionError,
    SteganographyError,
)
from app.schemas.encryption import EncryptedPayload
from app.services.adaptive_steganography_service import AdaptiveSteganographyService
from app.services.encryption_service import EncryptionService

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


def _raw_rgb(image: Image.Image | bytes) -> bytes:
    """Return the raw RGB bytes of a PIL image or a PNG/JPEG byte string."""
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    return image.convert("RGB").tobytes()


def _png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_with_raw_changes(png: bytes, changes: dict[int, int]) -> bytes:
    """Return a PNG whose raw RGB bytes have been overwritten per `changes`."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    raw = bytearray(image.tobytes())
    for byte_index, value in changes.items():
        raw[byte_index] = value
    out = Image.frombytes("RGB", image.size, bytes(raw))
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_with_bit_flipped(png: bytes, byte_index: int) -> bytes:
    """Return a PNG with the LSB of raw RGB `byte_index` flipped."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    raw = bytearray(image.tobytes())
    raw[byte_index] ^= 0x01
    return _png_with_raw_changes(png, {byte_index: raw[byte_index]})


@pytest.fixture
def png_image() -> bytes:
    return _make_png()


@pytest.fixture
def adaptive_service() -> AdaptiveSteganographyService:
    return AdaptiveSteganographyService()


@pytest.fixture
def encryption_service() -> EncryptionService:
    return EncryptionService()


class TestRoundTrip:
    """Exact byte round trips through embed/extract."""

    def test_small_payload(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        payload = b"hello adaptive stego"
        assert (
            adaptive_service.extract(adaptive_service.embed(image=png_image, payload=payload))
            == payload
        )

    def test_all_byte_values(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=_ALL_BYTES)
        assert adaptive_service.extract(stego) == _ALL_BYTES

    def test_larger_payload(self, adaptive_service: AdaptiveSteganographyService) -> None:
        image = _make_png((128, 128))
        payload = _ALL_BYTES * 8  # 2048 bytes
        stego = adaptive_service.embed(image=image, payload=payload)
        assert adaptive_service.extract(stego) == payload

    def test_empty_payload(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"")
        assert adaptive_service.extract(stego) == b""

    def test_payload_is_not_modified(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        payload = b"untouched-by-adaptive-stego"
        stego = adaptive_service.embed(image=png_image, payload=payload)
        assert adaptive_service.extract(stego) == payload

    def test_repeated_embed_extract(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        for payload in (b"a", b"bb", _ALL_BYTES, b""):
            assert (
                adaptive_service.extract(adaptive_service.embed(image=png_image, payload=payload))
                == payload
            )

    def test_embedding_actually_changes_the_image(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"x" * 100)
        assert stego != png_image


class TestDeterminism:
    """Same inputs always produce identical output; extraction is stable."""

    def test_embed_is_deterministic(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        first = adaptive_service.embed(image=png_image, payload=b"data")
        second = adaptive_service.embed(image=png_image, payload=b"data")
        assert first == second

    def test_embed_deterministic_with_custom_weights(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        kwargs = {"edge_weight": 2.0, "texture_weight": 0.5}
        first = adaptive_service.embed(image=png_image, payload=b"data", **kwargs)
        second = adaptive_service.embed(image=png_image, payload=b"data", **kwargs)
        assert first == second

    def test_extract_is_deterministic(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=_ALL_BYTES)
        assert adaptive_service.extract(stego) == adaptive_service.extract(stego)

    def test_analysis_input_is_stable(self, png_image: bytes) -> None:
        """Re-analyzing the same image yields the same ranked order, so the
        embedding order derived from it is reproducible."""
        image = Image.open(io.BytesIO(png_image))
        assert embed_bytes(image, b"x") == embed_bytes(image, b"x")


class TestAdaptiveBehavior:
    """Bits must land in the highest-complexity region first."""

    def test_payload_bits_prefer_textured_region(
        self, adaptive_service: AdaptiveSteganographyService
    ) -> None:
        """With a smooth left half and textured right half, a small payload
        must be embedded only in the textured half (plus the fixed header)."""
        width, height = 64, 64
        image = _half_smooth_half_textured((width, height))
        payload = b"Z" * 300  # 2400 bits, well within the textured half's capacity
        stego = adaptive_service.embed(image=_png_bytes(image), payload=payload)

        cover_raw = _raw_rgb(image)
        stego_raw = _raw_rgb(stego)
        changed = [i for i in range(len(cover_raw)) if cover_raw[i] != stego_raw[i]]
        assert changed  # something was embedded

        textured_hits = 0
        for position in changed:
            if position < HEADER_BITS:
                continue  # the fixed header lives in the leading LSBs
            col = (position // 3) % width
            assert col >= width // 2  # every payload bit lands in the textured half
            textured_hits += 1
        assert textured_hits > 0

    def test_smooth_image_uses_row_major_order(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        """A flat image ties all scores at 0, so ranking degenerates to the
        row-major order — payload fills the positions right after the header."""
        payload = b"row-major-check" * 10
        stego = adaptive_service.embed(image=png_image, payload=payload)
        cover_raw = _raw_rgb(png_image)
        stego_raw = _raw_rgb(stego)
        boundary = HEADER_BITS + len(payload) * 8
        # Every LSB at/after the boundary must be untouched.
        assert stego_raw[boundary:] == cover_raw[boundary:]
        # ...and at least the header region changed.
        assert stego_raw[:HEADER_BITS] != cover_raw[:HEADER_BITS]

    def test_uses_phase25_analysis(self, png_image: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
        """embed_bytes drives Phase 2.5's analyze() with the caller's weights on
        the LSB-cleared image, and extract_bytes does the same on the stego."""
        image = Image.open(io.BytesIO(png_image))
        calls: list[tuple[bytes, tuple[int, int], float, float]] = []
        original = adaptive_embedding.analyze

        def spy(
            candidate: Image.Image,
            *,
            edge_weight: float = 1.0,
            texture_weight: float = 1.0,
        ) -> object:
            calls.append((candidate.tobytes(), candidate.size, edge_weight, texture_weight))
            return original(candidate, edge_weight=edge_weight, texture_weight=texture_weight)

        monkeypatch.setattr(adaptive_embedding, "analyze", spy)
        stego = embed_bytes(image, b"phase-25-in-the-loop", edge_weight=1.5, texture_weight=2.5)
        extract_bytes(stego)

        assert len(calls) == 2  # one analysis for embed, one for extract
        for candidate_bytes, _size, edge_weight, texture_weight in calls:
            # Analysis input is the image with every channel LSB cleared.
            assert all(byte & 1 == 0 for byte in candidate_bytes)
            # Weights are the header-quantized values (1.5 and 2.5 are exact).
            assert edge_weight == pytest.approx(1.5)
            assert texture_weight == pytest.approx(2.5)


class TestCapacity:
    """Capacity is validated; payloads are never silently truncated."""

    def test_max_payload_bytes_reports_capacity(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        expected = (64 * 64 * 3 - HEADER_BITS) // 8
        assert max_payload_bytes(image) == expected

    def test_exact_max_payload_fits(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        image = Image.open(io.BytesIO(png_image))
        payload = b"m" * max_payload_bytes(image)
        stego = adaptive_service.embed(image=png_image, payload=payload)
        assert adaptive_service.extract(stego) == payload

    def test_one_byte_over_capacity_rejected(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        image = Image.open(io.BytesIO(png_image))
        payload = b"m" * (max_payload_bytes(image) + 1)
        with pytest.raises(AdaptiveEmbeddingError, match="capacity"):
            adaptive_service.embed(image=png_image, payload=payload)

    def test_tiny_image_rejected(self, adaptive_service: AdaptiveSteganographyService) -> None:
        with pytest.raises(AdaptiveEmbeddingError):
            adaptive_service.embed(image=_make_png((1, 1)), payload=b"anything")


class TestImageIntegrity:
    """The input image is never mutated and the output stays a usable PNG."""

    def test_embed_does_not_mutate_input(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        before = image.tobytes()
        embed_bytes(image, b"hello")
        assert image.tobytes() == before  # returns a new image, input untouched

    def test_extract_does_not_mutate_input(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"data")
        image = Image.open(io.BytesIO(stego))
        before = image.tobytes()
        adaptive_service.extract(stego)
        assert image.tobytes() == before

    def test_output_is_readable_png(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"hidden data")
        image = Image.open(io.BytesIO(stego))
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (64, 64)

    def test_dimensions_unchanged(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=_ALL_BYTES)
        assert Image.open(io.BytesIO(stego)).size == (64, 64)

    def test_grayscale_and_rgba_normalize_consistently(
        self, adaptive_service: AdaptiveSteganographyService
    ) -> None:
        payload = b"rgb-normalization-check"
        for mode, color in (("L", 128), ("RGBA", (10, 20, 30, 128))):
            image = Image.new(mode, (64, 64), color=color)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            stego = adaptive_service.embed(image=buffer.getvalue(), payload=payload)
            assert adaptive_service.extract(stego) == payload
            assert Image.open(io.BytesIO(stego)).mode == "RGB"


class TestHeaderIntegrity:
    """Extraction must reject images without a valid, sane adaptive header."""

    def test_plain_image_has_no_payload(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        with pytest.raises(AdaptiveEmbeddingError, match="No valid adaptive payload"):
            adaptive_service.extract(png_image)

    def test_corrupted_magic_rejected(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"secret")
        corrupted = _png_with_bit_flipped(stego, byte_index=0)  # first magic bit
        with pytest.raises(AdaptiveEmbeddingError, match="No valid adaptive payload"):
            adaptive_service.extract(corrupted)

    def test_unsupported_version_rejected(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"secret")
        # Header byte 4 is the version (1); flipping its LSB yields version 0.
        corrupted = _png_with_bit_flipped(stego, byte_index=4 * 8)
        with pytest.raises(AdaptiveEmbeddingError, match="Unsupported adaptive payload version"):
            adaptive_service.extract(corrupted)

    def test_unsupported_flags_rejected(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"secret")
        # Header byte 5 is the flags field (must be 0); setting it to 1 is a
        # malformed/unknown metadata combination.
        corrupted = _png_with_raw_changes(stego, {5 * 8: 1})
        with pytest.raises(AdaptiveEmbeddingError, match="Unsupported adaptive payload flags"):
            adaptive_service.extract(corrupted)

    def test_zeroed_weights_rejected(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"secret")
        # Header bytes 6-9 hold edge_weight and texture_weight (uint16 each).
        corrupted = _png_with_raw_changes(stego, dict.fromkeys(range(6 * 8, 10 * 8), 0))
        with pytest.raises(AdaptiveEmbeddingError, match="weights are both zero"):
            adaptive_service.extract(corrupted)

    def test_overflowing_weight_rejected(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"secret")
        # Force edge_weight's uint16 to 0xFFFF (way above the 64.0 cap) by
        # setting the LSBs of header bytes 6-7.
        corrupted = _png_with_raw_changes(stego, dict.fromkeys(range(6 * 8, 8 * 8), 1))
        with pytest.raises(AdaptiveEmbeddingError, match="out of range"):
            adaptive_service.extract(corrupted)

    def test_oversized_length_rejected(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"small")
        # Header byte 10 is the length field's most significant byte; flipping
        # its LSB claims a multi-gigabyte payload that can never fit.
        corrupted = _png_with_bit_flipped(stego, byte_index=10 * 8)
        with pytest.raises(AdaptiveEmbeddingError, match="exceeds image capacity"):
            adaptive_service.extract(corrupted)

    def test_tiny_image_has_no_payload(
        self, adaptive_service: AdaptiveSteganographyService
    ) -> None:
        # 1x1 RGB = 3 bytes = 24 bits < 112-bit header.
        with pytest.raises(AdaptiveEmbeddingError, match="too small"):
            adaptive_service.extract(_make_png((1, 1)))


class TestMalformedInput:
    """Malformed or non-PNG input fails closed with AdaptiveEmbeddingError."""

    def test_empty_image_rejected(self, adaptive_service: AdaptiveSteganographyService) -> None:
        with pytest.raises(AdaptiveEmbeddingError, match="empty"):
            adaptive_service.embed(image=b"", payload=b"data")

    def test_non_image_bytes_rejected(self, adaptive_service: AdaptiveSteganographyService) -> None:
        with pytest.raises(AdaptiveEmbeddingError, match="Invalid image"):
            adaptive_service.embed(image=b"this is not an image at all", payload=b"data")

    def test_non_png_rejected(self, adaptive_service: AdaptiveSteganographyService) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buffer, format="JPEG")
        jpeg = buffer.getvalue()
        with pytest.raises(AdaptiveEmbeddingError, match="Only PNG"):
            adaptive_service.embed(image=jpeg, payload=b"data")
        with pytest.raises(AdaptiveEmbeddingError, match="Only PNG"):
            adaptive_service.extract(jpeg)

    def test_truncated_png_rejected(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"hidden data")
        with pytest.raises(AdaptiveEmbeddingError, match="Invalid image"):
            adaptive_service.extract(stego[: len(stego) // 2])


class TestPrimitives:
    """Low-level core.adaptive_embedding behavior."""

    def test_embed_extract_round_trip(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        payload = _ALL_BYTES
        stego = embed_bytes(image, payload)
        assert extract_bytes(stego) == payload

    def test_header_layout(self) -> None:
        assert len(ADAPTIVE_MAGIC) == 4
        assert FORMAT_VERSION == 1
        assert HEADER_SIZE == 14
        assert HEADER_BITS == HEADER_SIZE * 8

    def test_magic_is_distinct_from_phase24(self) -> None:
        """Adaptive and sequential payloads must never cross-parse."""
        assert ADAPTIVE_MAGIC != phase24.MAGIC

    def test_non_bytes_payload_rejected(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        with pytest.raises(TypeError, match="bytes"):
            embed_bytes(image, "not bytes")  # type: ignore[arg-type]

    def test_invalid_weights_rejected(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        with pytest.raises(ValueError, match="non-negative"):
            embed_bytes(image, b"x", edge_weight=-1.0)
        with pytest.raises(ValueError, match="both be zero"):
            embed_bytes(image, b"x", edge_weight=0.0, texture_weight=0.0)
        with pytest.raises(ValueError, match="must not exceed"):
            embed_bytes(image, b"x", edge_weight=65.0)


class TestFormatSeparation:
    """Phase 2.6 and Phase 2.4 payloads are mutually invisible."""

    def test_adaptive_payload_not_readable_by_phase24(
        self, adaptive_service: AdaptiveSteganographyService, png_image: bytes
    ) -> None:
        stego = adaptive_service.embed(image=png_image, payload=b"adaptive secret")
        image = Image.open(io.BytesIO(stego))
        with pytest.raises(SteganographyError, match="No valid BitwiseHide payload"):
            phase24.extract_bytes(image)

    def test_phase24_payload_not_readable_by_adaptive(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        phase24_stego = phase24.embed_bytes(image, b"sequential secret")
        with pytest.raises(AdaptiveEmbeddingError, match="No valid adaptive payload"):
            extract_bytes(phase24_stego)


class TestPhase23Integration:
    """End-to-end: EncryptionService -> AdaptiveSteganographyService -> decryption."""

    def test_encrypted_payload_full_round_trip(
        self,
        adaptive_service: AdaptiveSteganographyService,
        encryption_service: EncryptionService,
        png_image: bytes,
    ) -> None:
        plaintext = "phase-2.6-integration-secret"
        payload = encryption_service.encrypt(plaintext=plaintext, password=PASSWORD)
        stego = adaptive_service.embed(image=png_image, payload=payload.to_bytes())

        extracted = adaptive_service.extract(stego)
        restored = EncryptedPayload.from_bytes(extracted)
        assert encryption_service.decrypt(restored, PASSWORD) == plaintext

    def test_wrong_password_fails(
        self,
        adaptive_service: AdaptiveSteganographyService,
        encryption_service: EncryptionService,
        png_image: bytes,
    ) -> None:
        payload = encryption_service.encrypt(plaintext="secret", password=PASSWORD)
        stego = adaptive_service.embed(image=png_image, payload=payload.to_bytes())

        extracted = adaptive_service.extract(stego)
        restored = EncryptedPayload.from_bytes(extracted)
        with pytest.raises(EncryptionError):
            encryption_service.decrypt(restored, WRONG_PASSWORD)

    def test_corrupted_encrypted_payload_fails_closed(
        self,
        adaptive_service: AdaptiveSteganographyService,
        encryption_service: EncryptionService,
        png_image: bytes,
    ) -> None:
        """A bit flip in the encrypted payload region must never yield plaintext."""
        plaintext = "corruption-sensitive"
        payload = encryption_service.encrypt(plaintext=plaintext, password=PASSWORD)
        original = payload.to_bytes()
        stego = adaptive_service.embed(image=png_image, payload=original)

        # The smooth PNG ranks in row-major order, so the first payload byte
        # lives at stream position HEADER_BITS. Flipping it corrupts the JSON.
        corrupted = _png_with_bit_flipped(stego, byte_index=HEADER_BITS)
        extracted = adaptive_service.extract(corrupted)
        assert extracted != original

        # Parsing or decryption must fail — corrupted plaintext is never returned.
        try:
            restored = EncryptedPayload.from_bytes(extracted)
        except ValueError:
            return  # malformed JSON failed closed at parse time
        with pytest.raises(EncryptionError):
            encryption_service.decrypt(restored, PASSWORD)
