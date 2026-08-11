"""
Phase 2.8.2 label-generation tests.

Covers:
- the additive structural_similarity_map refactor: its mean equals
  structural_similarity(...), its shape matches the input (H, W) / (H, W, 3),
  and it fails closed on invalid arguments
- ai.labels.suitability_label_map: HxW shape, [0, 1] range, finiteness,
  determinism and repeated-call identity
- the closed form is EXACTLY the single-pixel perturbation probe: the label
  equals structural_similarity_map(Z, Z + E_p) at every pixel, including the
  reflect-padding border band and clamped small-image windows
- smooth-vs-textured separability: the marginal cost (1 - label) of a smooth
  region exceeds a textured region's by orders of magnitude
- iso-luminant RGB color texture is captured per-channel, not flattened by
  grayscale luma
- reflect-padding border behavior: a solid image's interior is uniform, while
  pixels one in from each corner (whose spike mirrors into its own window)
  score measurably lower
- synthetic smooth, textured, and RGB/color-texture images
- grayscale/RGBA normalization and fail-closed invalid input

No external services and no ML framework are involved; all images are
generated in memory.
"""

from __future__ import annotations

import numpy as np
import pytest
from ai.labels import suitability_label_map
from PIL import Image

from app.core.evaluation import (
    image_to_rgb_array,
    structural_similarity,
    structural_similarity_map,
)


def _image_from_pixels(rows: list[list[tuple[int, int, int]]]) -> Image.Image:
    """Build a small RGB PIL image from literal rows of (R, G, B) pixels."""
    width = len(rows[0])
    height = len(rows)
    image = Image.new("RGB", (width, height))
    image.putdata([pixel for row in rows for pixel in row])
    return image


def _solid_image(
    size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (128, 128, 128)
) -> Image.Image:
    """Create a solid-color (smooth) RGB image."""
    return Image.new("RGB", size, color=color)


def _checkerboard(
    size: tuple[int, int],
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    tile: int = 2,
) -> Image.Image:
    """Create an RGB checkerboard alternating between two colors."""
    width, height = size
    image = Image.new("RGB", size)
    image.putdata(
        [colors[((x // tile) + (y // tile)) % 2] for y in range(height) for x in range(width)]
    )
    return image


def _half_smooth_half_textured(size: tuple[int, int]) -> Image.Image:
    """Left half is flat gray, right half is a 0/255 checkerboard."""
    width, height = size
    image = _solid_image(size, color=(128, 128, 128))
    checker = _checkerboard((width - width // 2, height), ((0, 0, 0), (255, 255, 255)), tile=1)
    image.paste(checker, (width // 2, 0))
    return image


def _even_rgb_array(size: tuple[int, int]) -> np.ndarray:
    """Deterministic RGB uint8 array whose every channel is LSB-cleared (even).

    LSB-cleared channels make Image.fromarray(...) reproduce the exact analysis
    domain `Z` the label is defined on, so the brute-force probe compares like
    for like.
    """
    height, width = size
    rng = np.random.default_rng(7)
    return (rng.integers(0, 128, size=(height, width, 3), dtype=np.uint8) * 2).astype(np.uint8)


def _brute_single_pixel_probe(z: np.ndarray) -> np.ndarray:
    """Exact per-pixel SSIM map of Z vs Z + E_p, one SSIM call per pixel.

    `z` must be LSB-cleared (every channel even) so the +1 flip never wraps
    past 255. This is the ground truth the closed-form label must reproduce.
    """
    height, width, _ = z.shape
    probe = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            perturbed = z.copy()
            perturbed[y, x, :] = z[y, x, :] + 1
            probe[y, x] = structural_similarity_map(z, perturbed)[y, x].mean()
    return probe


# --- Structural-similarity map refactor ---------------------------------------


class TestStructuralSimilarityMap:
    """structural_similarity_map exposes the per-pixel map the scalar metric uses."""

    def test_map_mean_equals_scalar_metric_rgb(self) -> None:
        cover = _image_from_pixels([[(x, y, 3) for x in range(8)] for y in range(8)])
        stego = _image_from_pixels([[(x, y + 1, 3) for x in range(8)] for y in range(8)])
        cover_arr = image_to_rgb_array(cover)
        stego_arr = image_to_rgb_array(stego)
        ssim_map = structural_similarity_map(cover_arr, stego_arr)
        assert ssim_map.shape == (8, 8, 3)
        assert float(np.mean(ssim_map)) == pytest.approx(
            structural_similarity(cover_arr, stego_arr)
        )

    def test_map_mean_equals_scalar_metric_gray(self) -> None:
        cover = _image_from_pixels([[(x, y, 3) for x in range(8)] for y in range(8)])
        stego = _image_from_pixels([[(x, y + 1, 3) for x in range(8)] for y in range(8)])
        cover_gray = image_to_rgb_array(cover)[:, :, 0]
        stego_gray = image_to_rgb_array(stego)[:, :, 0]
        ssim_map = structural_similarity_map(cover_gray, stego_gray)
        assert ssim_map.shape == (8, 8)
        assert float(np.mean(ssim_map)) == pytest.approx(
            structural_similarity(cover_gray, stego_gray)
        )

    def test_map_rejects_mismatched_shapes(self) -> None:
        cover = np.zeros((4, 4, 3), dtype=np.uint8)
        stego = np.zeros((4, 5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="identical shapes"):
            structural_similarity_map(cover, stego)

    def test_map_rejects_invalid_arguments(self) -> None:
        cover = np.zeros((4, 4, 3), dtype=np.uint8)
        stego = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="data_range"):
            structural_similarity_map(cover, stego, data_range=0.0)
        with pytest.raises(ValueError, match="window_size"):
            structural_similarity_map(cover, stego, window_size=8)


# --- Suitability label map ----------------------------------------------------


class TestSuitabilityLabelMap:
    """suitability_label_map produces a deterministic HxW [0, 1] map."""

    def test_output_shape(self) -> None:
        square = _solid_image((64, 64))
        non_square = _solid_image((40, 30))
        assert suitability_label_map(square).shape == (64, 64)
        assert suitability_label_map(non_square).shape == (30, 40)

    def test_values_within_unit_range(self) -> None:
        labels = suitability_label_map(_checkerboard((32, 32), ((0, 0, 0), (255, 255, 255))))
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_deterministic(self) -> None:
        image = _checkerboard((32, 32), ((0, 0, 0), (255, 255, 255)))
        first = suitability_label_map(image)
        second = suitability_label_map(image)
        assert np.array_equal(first, second)

    def test_repeated_calls_identical(self) -> None:
        image = _half_smooth_half_textured((64, 64))
        results = [suitability_label_map(image) for _ in range(3)]
        assert all(np.array_equal(results[0], result) for result in results[1:])

    def test_finite(self) -> None:
        labels = suitability_label_map(_checkerboard((32, 32), ((0, 0, 0), (255, 255, 255))))
        assert np.isfinite(labels).all()

    def test_smooth_image_scores_high(self) -> None:
        """A solid image scores high everywhere (suitability close to 1)."""
        labels = suitability_label_map(_solid_image((64, 64), color=(128, 128, 128)))
        assert labels.shape == (64, 64)
        assert (labels > 0.99).all()
        assert labels.mean() > 0.998

    def test_textured_image(self) -> None:
        """A high-frequency checkerboard preserves structure under the +1 probe,
        so the label map stays high and well-formed."""
        labels = suitability_label_map(_checkerboard((32, 32), ((0, 0, 0), (255, 255, 255))))
        assert labels.shape == (32, 32)
        assert np.isfinite(labels).all()
        assert (labels >= 0.0).all() and (labels <= 1.0).all()
        assert (labels > 0.99).all()

    def test_color_texture_image(self) -> None:
        """A red/green checkerboard is a color texture that grayscale flattens
        (nearly iso-luminant); the RGB-based label still handles it."""
        colors = ((255, 0, 0), (0, 128, 0))  # luma ~76 vs ~75 in grayscale
        labels = suitability_label_map(_checkerboard((32, 32), colors))
        assert labels.shape == (32, 32)
        assert np.isfinite(labels).all()
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_grayscale_and_rgba_normalize_consistently(self) -> None:
        for mode, color in (("L", 128), ("RGBA", (10, 20, 30, 128))):
            image = Image.new(mode, (32, 32), color=color)
            labels = suitability_label_map(image)
            assert labels.shape == (32, 32)
            assert np.isfinite(labels).all()

    def test_non_pil_input_raises(self) -> None:
        with pytest.raises(ValueError, match="PIL image"):
            suitability_label_map("not an image")  # type: ignore[arg-type]

    def test_empty_image_raises(self) -> None:
        with pytest.raises(ValueError, match="positive width and height"):
            suitability_label_map(Image.new("RGB", (0, 8)))


# --- Exactness against the single-pixel perturbation probe --------------------


class TestExactProbeEquivalence:
    """The closed-form label is exactly the single-pixel perturbation probe."""

    @pytest.mark.parametrize("size", [(9, 12), (12, 12)])
    def test_closed_form_equals_single_pixel_probe(self, size: tuple[int, int]) -> None:
        z = _even_rgb_array(size)
        label = suitability_label_map(Image.fromarray(z))
        probe = _brute_single_pixel_probe(z)
        assert label.shape == size
        # The closed form reproduces the probe to machine precision at every
        # pixel (borders included); assert at 1e-9 for float headroom.
        assert np.allclose(label, probe, rtol=0.0, atol=1e-9)

    def test_closed_form_equals_probe_on_smooth_image(self) -> None:
        z = np.full((8, 9, 3), 128, dtype=np.uint8)  # 128 is LSB-cleared
        label = suitability_label_map(Image.fromarray(z))
        probe = _brute_single_pixel_probe(z)
        assert np.allclose(label, probe, rtol=0.0, atol=1e-9)


# --- Smooth-vs-textured separability ------------------------------------------


class TestSeparability:
    def test_smooth_region_cost_dwarfs_textured(self) -> None:
        size = (64, 64)
        labels = suitability_label_map(_half_smooth_half_textured(size))
        width = size[0]
        smooth_mean = labels[:, : width // 2].mean()
        textured_mean = labels[:, width // 2 :].mean()
        # Textured pixels tolerate the write better than smooth pixels.
        assert textured_mean > smooth_mean
        # Marginal cost 1 - label is an order-of-magnitude measure apart.
        smooth_cost = 1.0 - smooth_mean
        textured_cost = 1.0 - textured_mean
        assert smooth_cost / textured_cost > 100.0


# --- Color texture beyond grayscale luma -------------------------------------


class TestColorTexture:
    def test_iso_luminant_color_texture_captured_per_channel(self) -> None:
        # Red (255,0,0) vs green (0,128,0): grayscale luma 76.2 vs 75.1, so a
        # luma flatten sees near-flat content. Per-channel RGB variance keeps
        # the color texture visible, so the label must score it higher than
        # its own grayscale flatten does.
        colors = ((255, 0, 0), (0, 128, 0))
        color_image = _checkerboard((32, 32), colors)
        color_labels = suitability_label_map(color_image)
        gray_image = Image.fromarray(np.asarray(color_image.convert("L"), dtype=np.uint8))
        gray_labels = suitability_label_map(gray_image.convert("RGB"))
        assert color_labels.shape == (32, 32)
        assert np.isfinite(color_labels).all()
        assert color_labels.min() >= 0.0
        assert color_labels.max() <= 1.0
        assert color_labels.mean() > gray_labels.mean() + 2e-4


# --- Reflect-padding border behavior -----------------------------------------


class TestBorderBehavior:
    def test_solid_interior_is_uniform_and_reflect_band_is_lower(self) -> None:
        labels = suitability_label_map(_solid_image((64, 64), color=(128, 128, 128)))
        interior = labels[5:-5, 5:-5]
        # Reflect-free interior: every pixel is identical.
        assert np.ptp(interior) < 1e-12
        # The exact corner (0, 0) has no mirror image, so it equals interior.
        assert labels[0, 0] == pytest.approx(float(interior.mean()))
        # One pixel in from each corner the spike mirrors into its own window,
        # so the effective write is heavier and the label drops.
        assert labels[1, 1] < float(interior.mean())
