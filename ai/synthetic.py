"""
Deterministic synthetic RGB image generation (Phase 2.8.1, training-time only).

These builders produce the reproducible in-memory textures the dataset layer
uses for offline smoke tests and local experimentation: a user can prepare and
split a small synthetic dataset without ever downloading BSDS500. Every builder
is a pure function of its arguments — the same arguments always yield the same
image bytes, and no builder touches the network.

Kinds and why they matter for the suitability model
---------------------------------------------------
- smooth: flat solid regions. These are where a single +-1 LSB flip is most
  visible, so the learned label must mark them least suitable.
- checkerboard: high-frequency binary texture. Flips hide well here; the label
  must mark them most suitable.
- noise: fully unstructured per-channel noise, the other extreme of texture.
- gradient: smoothly varying luminance. A mid-texture case whose local
  contrast rises with the gradient slope.
- color_texture: an RGB checkerboard whose two colors are nearly iso-luminant
  in grayscale (red vs a dark green, luma ~76 vs ~75). A luma-based flatten
  sees near-flat content, so only a per-channel RGB treatment preserves the
  texture — the exact failure mode the Phase 2.8.2 label must not repeat.

Design constraints:
- Deterministic: noise uses numpy's seeded Generator (default_rng(seed)), every
  other kind is a closed-form pixel pattern.
- No network, no filesystem, no ML framework: builders return PIL Images.
- Invalid *kind* raises ValueError; invalid kind-specific arguments (e.g. a bad
  gradient axis) raise ValueError — mirroring the caller-error split elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

#: The canonical catalogue of kinds, in the order they cycle through a
#: synthetic dataset: smooth, checkerboard, noise, gradient, color texture.
SYNTHETIC_KINDS: tuple[str, ...] = (
    "smooth",
    "checkerboard",
    "noise",
    "gradient",
    "color_texture",
)

#: Default black/white checkerboard pair.
_BLACK_WHITE = ((0, 0, 0), (255, 255, 255))

#: Nearly iso-luminant pair: grayscale luma ~76 (red) vs ~75 (green).
_ISO_LUMINANT = ((255, 0, 0), (0, 128, 0))


def smooth_image(
    *,
    size: tuple[int, int] = (64, 64),
    color: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """A solid-color image — the canonical flat/smooth region."""
    return Image.new("RGB", size, color=color)


def checkerboard_image(
    *,
    size: tuple[int, int] = (64, 64),
    colors: tuple[tuple[int, int, int], tuple[int, int, int]] = _BLACK_WHITE,
    tile: int = 2,
) -> Image.Image:
    """An RGB checkerboard alternating between `colors` every `tile` pixels."""
    width, height = size
    image = Image.new("RGB", size)
    image.putdata(
        [colors[((x // tile) + (y // tile)) % 2] for y in range(height) for x in range(width)]
    )
    return image


def noise_image(
    *,
    size: tuple[int, int] = (64, 64),
    seed: int = 0,
    low: int = 0,
    high: int = 255,
) -> Image.Image:
    """Uniform per-channel noise in ``[low, high]``, deterministic in `seed`.

    `size` is PIL's ``(width, height)`` convention, matching every other
    builder.
    """
    width, height = size
    rng = np.random.default_rng(seed)
    arr = rng.integers(low, high + 1, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def gradient_image(
    *,
    size: tuple[int, int] = (64, 64),
    axis: str = "horizontal",
    start: tuple[int, int, int] = (0, 0, 0),
    end: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """A smooth linear RGB gradient from `start` to `end` along `axis`.

    Each row (horizontal) or column (vertical) is constant; the perpendicular
    axis ramps monotonically. The gradient is the mid-texture reference case:
    flips are visible in the flat tail and hidden in the steep head.
    `size` is PIL's ``(width, height)`` convention.

    Raises:
        ValueError: If `axis` is not ``"horizontal"`` or ``"vertical"``.
    """
    if axis not in ("horizontal", "vertical"):
        raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis!r}.")
    width, height = size
    length = width if axis == "horizontal" else height
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64)
    end_arr = np.asarray(end, dtype=np.float64)
    ramp = start_arr[None, :] + t[:, None] * (end_arr[None, :] - start_arr[None, :])
    if axis == "horizontal":
        arr = np.broadcast_to(ramp[None, :, :], (height, width, 3))
    else:
        arr = np.broadcast_to(ramp[:, None, :], (height, width, 3))
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), mode="RGB")


def color_texture_image(
    *,
    size: tuple[int, int] = (64, 64),
    colors: tuple[tuple[int, int, int], tuple[int, int, int]] = _ISO_LUMINANT,
    tile: int = 2,
) -> Image.Image:
    """An RGB checkerboard whose colors are nearly iso-luminant in grayscale.

    Grayscale flattening reads this as near-flat content; a per-channel RGB
    treatment must still recover the texture. This is the regression target for
    RGB-aware suitability labels.
    """
    return checkerboard_image(size=size, colors=colors, tile=tile)


_BUILDERS: dict[str, Callable[..., Image.Image]] = {
    "smooth": smooth_image,
    "checkerboard": checkerboard_image,
    "noise": noise_image,
    "gradient": gradient_image,
    "color_texture": color_texture_image,
}


def synthetic_image(kind: str, **kwargs: object) -> Image.Image:
    """Render a deterministic synthetic RGB image of `kind`.

    Args:
        kind: One of SYNTHETIC_KINDS.
        kwargs: Forwarded to the kind's builder (size, seed, colors, ...).

    Returns:
        An RGB PIL image of the requested kind.

    Raises:
        ValueError: If `kind` is not a known synthetic kind.
    """
    builder = _BUILDERS.get(kind)
    if builder is None:
        expected = ", ".join(SYNTHETIC_KINDS)
        raise ValueError(f"unknown synthetic kind {kind!r}; expected one of {expected}.")
    return builder(**kwargs)
