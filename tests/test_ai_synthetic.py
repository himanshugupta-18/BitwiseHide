"""
Phase 2.8.1 synthetic-image generation tests.

Covers ai.synthetic's deterministic, network-free RGB builders:
- every kind renders a correctly-shaped RGB image
- the synthetic_image dispatcher (unknown kind raises ValueError)
- smooth constancy, checkerboard alternation, noise determinism
- gradient monotonicity along the chosen axis and constancy across it
- iso-luminant color texture: two distinct RGB colors whose grayscale lumas
  are nearly equal (the case grayscale flattening cannot see as texture)

No external services are involved; all images are generated in memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

import numpy as np
import pytest

from ai.synthetic import (
    SYNTHETIC_KINDS,
    checkerboard_image,
    color_texture_image,
    gradient_image,
    noise_image,
    smooth_image,
    synthetic_image,
)


def _as_array(image: Image.Image) -> np.ndarray:
    """Normalize a synthetic image to a uint8 RGB array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


class TestSyntheticKinds:
    def test_all_kinds_render_rgb(self) -> None:
        for kind in SYNTHETIC_KINDS:
            image = synthetic_image(kind, size=(16, 24))
            assert image.mode == "RGB"
            assert image.size == (16, 24)
            arr = _as_array(image)
            assert arr.shape == (24, 16, 3)

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown synthetic kind"):
            synthetic_image("does-not-exist")

    def test_kind_catalogue_has_expected_examples(self) -> None:
        assert set(SYNTHETIC_KINDS) == {
            "smooth",
            "checkerboard",
            "noise",
            "gradient",
            "color_texture",
        }


class TestSmoothImage:
    def test_constant_color(self) -> None:
        arr = _as_array(smooth_image(size=(8, 8), color=(10, 20, 30)))
        assert np.array_equal(arr, np.full((8, 8, 3), (10, 20, 30)))


class TestCheckerboardImage:
    def test_alternates_between_colors(self) -> None:
        colors = ((0, 0, 0), (255, 255, 255))
        arr = _as_array(checkerboard_image(size=(8, 8), colors=colors, tile=2))
        assert tuple(arr[0, 0]) == (0, 0, 0)
        assert tuple(arr[0, 2]) == (255, 255, 255)
        assert tuple(arr[2, 0]) == (255, 255, 255)
        assert tuple(arr[2, 2]) == (0, 0, 0)


class TestNoiseImage:
    def test_deterministic_in_seed(self) -> None:
        first = _as_array(noise_image(size=(16, 16), seed=3))
        second = _as_array(noise_image(size=(16, 16), seed=3))
        different = _as_array(noise_image(size=(16, 16), seed=4))
        assert np.array_equal(first, second)
        assert not np.array_equal(first, different)

    def test_values_stay_in_range(self) -> None:
        arr = _as_array(noise_image(size=(8, 8), low=10, high=20))
        assert arr.min() >= 10
        assert arr.max() <= 20


class TestGradientImage:
    def test_horizontal_ramps_along_x(self) -> None:
        arr = _as_array(
            gradient_image(size=(16, 8), axis="horizontal", start=(0, 0, 0), end=(255, 255, 255))
        )
        assert arr.shape == (8, 16, 3)
        # Every row is identical; the ramp is monotonic along x.
        assert np.array_equal(arr[0], arr[7])
        for x in range(1, 16):
            assert arr[0, x, 0] >= arr[0, x - 1, 0]
        assert arr[0, 0, 0] == 0
        assert arr[0, 15, 0] == 255

    def test_vertical_ramps_along_y(self) -> None:
        arr = _as_array(
            gradient_image(size=(8, 16), axis="vertical", start=(0, 0, 0), end=(255, 255, 255))
        )
        assert arr.shape == (16, 8, 3)
        assert np.array_equal(arr[:, 0], arr[:, 7])
        for y in range(1, 16):
            assert arr[y, 0, 0] >= arr[y - 1, 0, 0]
        assert arr[0, 0, 0] == 0
        assert arr[15, 0, 0] == 255

    def test_invalid_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            gradient_image(size=(8, 8), axis="diagonal")


class TestColorTextureImage:
    def test_two_iso_luminant_colors(self) -> None:
        image = color_texture_image(size=(16, 16))
        arr = _as_array(image)
        distinct = {tuple(px) for px in arr.reshape(-1, 3).tolist()}
        assert distinct == {(255, 0, 0), (0, 128, 0)}
        # Nearly iso-luminant in grayscale: luma flattening sees ~flat content.
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        lumas = set(gray.reshape(-1).tolist())
        assert len(lumas) == 2
        assert max(lumas) - min(lumas) <= 2
