"""
Deterministic adaptive embedding analysis for RGB/grayscale images.

This is the NON-AI baseline that scores every pixel by how visually complex its
neighborhood is — regions with strong edges or high local texture are preferred
embedding locations, smooth regions are avoided. The future CNN/ML embedding
selector is expected to replace or improve this scoring function; nothing here
is machine learning.

Architecture decisions:
- PURE analysis: this module produces a score map and a ranked list of embedding
  locations. It performs NO embedding (that stays in core.steganography) and NO
  encryption (that stays in core.crypto). It touches no auth, database, or API.
- DETERMINISTIC: the same image + same weights ALWAYS yields the same map. There
  is no random pixel selection and no hidden state.
- Per-pixel score = weighted average of two normalized components:
    1. Sobel edge magnitude  -> normalized by its theoretical maximum
       (4 * 255 * sqrt(2)) on 8-bit input.
    2. Local 3x3 variance   -> normalized by its theoretical maximum (127.5^2).
  Both components are computed on the grayscale ("L") conversion of the image,
  so RGB and pre-grayscaled input produce identical scores.
- Neighborhood sampling uses replicate borders (edge pixels clamp), so the map
  is defined at every pixel and its dimensions always match the input image.
- The input image is never mutated; only `convert` (a copy) and `tobytes` are
  used.
- Invalid *input* raises AdaptiveAnalysisError; invalid *weights* raise
  ValueError (they are caller misconfiguration, mirroring the scrypt-parameter
  validation in core.crypto).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from PIL import Image

from app.core.exceptions import AdaptiveAnalysisError

# 3x3 Sobel kernels for horizontal (x) and vertical (y) gradients.
_SOBEL_X: tuple[tuple[float, float, float], ...] = (
    (-1, 0, 1),
    (-2, 0, 2),
    (-1, 0, 1),
)
_SOBEL_Y: tuple[tuple[float, float, float], ...] = (
    (-1, -2, -1),
    (0, 0, 0),
    (1, 2, 1),
)
#: Maximum Sobel magnitude achievable on 8-bit pixel values (|gx|,|gy| <= 4*255).
_MAX_EDGE_MAGNITUDE = 4 * 255 * math.sqrt(2)
#: Maximum variance of an 8-bit-valued sample set (0/255 split), upper bound for
#: the 3x3 local-variance component.
_MAX_VARIANCE = 127.5**2
#: Default neighborhood window radius (3x3).
_WINDOW_RADIUS = 1


@dataclass(frozen=True)
class ComplexityMap:
    """
    Deterministic per-pixel complexity scores and ranked embedding locations.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        scores: Row-major ``height x width`` per-pixel suitability, each in
            ``[0.0, 1.0]`` (higher = more visually complex).
        ranked: One ``(row, col, score)`` entry per pixel, sorted by descending
            score (ties broken by row, then column) — the embedding priority.
    """

    width: int
    height: int
    scores: list[list[float]]
    ranked: list[tuple[int, int, float]]


def analyze(
    image: Image.Image,
    *,
    edge_weight: float = 1.0,
    texture_weight: float = 1.0,
) -> ComplexityMap:
    """
    Score every pixel of `image` for embedding suitability.

    Args:
        image: A PIL image (RGB, grayscale, RGBA, palette, etc. — all are
            converted to grayscale internally; the image is never modified).
        edge_weight: Relative importance of the Sobel edge component (>= 0).
        texture_weight: Relative importance of the local-variance component
            (>= 0). At least one weight must be positive; the final score is the
            normalized weighted average, so it always lies in ``[0.0, 1.0]``.

    Returns:
        A ComplexityMap with a per-pixel score in ``[0.0, 1.0]`` and a ranked
        list of embedding locations, ordered from most to least suitable.

    Raises:
        AdaptiveAnalysisError: If `image` is not a PIL image or has zero
            width/height.
        ValueError: If the weights are out of range (both negative or both zero).
    """
    if not isinstance(image, Image.Image):
        raise AdaptiveAnalysisError(message="Adaptive analysis requires a PIL image.")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise AdaptiveAnalysisError(message="Image must have positive width and height.")
    if edge_weight < 0 or texture_weight < 0 or edge_weight + texture_weight <= 0:
        msg = "Analysis weights must be non-negative and not both zero."
        raise ValueError(msg)

    pixels = list(image.convert("L").tobytes())
    weight_sum = edge_weight + texture_weight

    scores: list[list[float]] = []
    ranked: list[tuple[int, int, float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            gx, gy, variance = _local_features(pixels, width, height, x, y)
            edge_norm = math.hypot(gx, gy) / _MAX_EDGE_MAGNITUDE
            texture_norm = variance / _MAX_VARIANCE
            score = (edge_weight * edge_norm + texture_weight * texture_norm) / weight_sum
            score = min(1.0, max(0.0, score))
            row.append(score)
            ranked.append((y, x, score))
        scores.append(row)

    ranked.sort(key=lambda entry: (-entry[2], entry[0], entry[1]))
    return ComplexityMap(
        width=width,
        height=height,
        scores=scores,
        ranked=ranked,
    )


def _local_features(
    pixels: Sequence[int],
    width: int,
    height: int,
    x: int,
    y: int,
) -> tuple[float, float, float]:
    """
    Compute (gx, gy, variance) over the 3x3 neighborhood of pixel (x, y).

    Out-of-bounds neighbors replicate the edge pixel (clamp), so the result is
    defined for every pixel and the map keeps the input dimensions.
    """
    gx = 0.0
    gy = 0.0
    total = 0
    total_sq = 0
    count = 0
    for ky in range(-_WINDOW_RADIUS, _WINDOW_RADIUS + 1):
        py = min(max(y + ky, 0), height - 1)
        row_base = py * width
        for kx in range(-_WINDOW_RADIUS, _WINDOW_RADIUS + 1):
            px = min(max(x + kx, 0), width - 1)
            value = pixels[row_base + px]
            gx += value * _SOBEL_X[ky + _WINDOW_RADIUS][kx + _WINDOW_RADIUS]
            gy += value * _SOBEL_Y[ky + _WINDOW_RADIUS][kx + _WINDOW_RADIUS]
            total += value
            total_sq += value * value
            count += 1

    mean = total / count
    # Guard against tiny negative values from float rounding.
    variance = max(0.0, total_sq / count - mean * mean)
    return gx, gy, variance
