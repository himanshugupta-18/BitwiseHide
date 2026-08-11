"""
Deterministic image-quality metrics for steganography evaluation (Phase 2.7).

This is the pure, framework-agnostic metric layer. It computes objective
quality between a cover image and a stego image. It performs NO embedding
(that stays in core.steganography / core.adaptive_embedding), NO encryption
(core.crypto), and NO I/O beyond what PIL gives us — the service layer
(services.evaluation_service) owns PNG decoding and the embedding run.

Architecture decisions:
- PURE FUNCTIONS + dataclasses: every metric is deterministic; the same two
  arrays ALWAYS yield the same value. There is no randomness and no hidden
  state, mirroring core.adaptive.
- Array-based: metric functions operate on numpy uint8 arrays of shape
  (H, W) or (H, W, 3). ``image_to_rgb_array`` is the single PIL-to-RGB
  boundary, so every metric compares the SAME normalized RGB pixel values
  the embedders actually mutate.
- RGB-normalized inputs: both images are converted to 8-bit RGB before any
  comparison, so RGBA/pre-grayscaled covers and the RGB output of the stego
  services are always compared on the same representation.
- SSIM is implemented in pure numpy (Wang et al. 2004) with a Gaussian
  window and reflect-padded convolution, so NO scipy/scikit-image
  dependency is required. The window is clamped to the image size so the
  metric is defined at every pixel of even tiny images.
- Invalid *input* raises EvaluationError; invalid *arguments* (mismatched
  array shapes, non-positive data_range) raise ValueError — mirroring the
  input-vs-caller-error split in core.adaptive.

Metrics and why they matter for steganography
--------------------------------------------
MSE (mean squared error)
    The average squared difference between the original RGB pixel values and
    the stego RGB pixel values. 0 means the images are pixel-identical. It is
    the raw fidelity floor every other metric builds on, and it is exactly
    what a stego embedder trades away: every hidden bit changes a pixel value
    by +-1, so the payload raises MSE by one unit per flipped byte.

PSNR (peak signal-to-noise ratio)
    A logarithmic rescaling of MSE: PSNR = 10*log10(MAX^2 / MSE), with
    MAX = 255 for 8-bit images. Higher is better; infinity for identical
    images (MSE = 0). PSNR is the de-facto fidelity metric in image
    processing and makes small MSE differences readable in dB.

SSIM (structural similarity, Wang et al. 2004)
    Compares luminance, contrast, and structure over local Gaussian windows,
    in [-1, 1] with 1 meaning identical. Unlike MSE/PSNR it is
    perception-motivated: a +-1 change in a flat region is more visible than
    the same change in textured noise, and SSIM reflects that. For adaptive
    steganography this is the metric that should show WHERE a method wins —
    the whole point of Phase 2.5/2.6 is to place bits where they are least
    visible, and SSIM is the closest cheap proxy for "least visible."
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.core.exceptions import EvaluationError

# --- Defaults for the structural-similarity calculation (Wang et al. 2004) ---
_SSIM_WINDOW_SIZE = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03

#: Maximal 8-bit pixel value — the correct MAX for PSNR over uint8 RGB.
MAX_PIXEL_VALUE = 255.0

#: A uint8 array of shape (H, W) or (H, W, 3).
UInt8Array = NDArray[np.uint8]
#: A float64 array used for arithmetic (same shapes as UInt8Array).
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImageQualityMetrics:
    """
    Objective quality of a stego image relative to its cover.

    Attributes:
        mse: Mean squared error over all RGB pixels (0 == identical).
        psnr: Peak signal-to-noise ratio in dB, derived from MSE;
            ``math.inf`` when MSE is zero (identical images).
        ssim: Structural similarity in ``[-1, 1]`` (1 == identical).
    """

    mse: float
    psnr: float
    ssim: float


def image_to_rgb_array(image: Image.Image) -> UInt8Array:
    """
    Normalize any PIL image to a uint8 RGB array of shape (H, W, 3).

    Raises:
        EvaluationError: If `image` is not a PIL image or is empty.
    """
    if not isinstance(image, Image.Image):
        msg = f"Evaluation requires a PIL image, got {type(image).__name__}."
        raise EvaluationError(message=msg)
    if image.size[0] <= 0 or image.size[1] <= 0:
        msg = f"Image must have positive width and height, got {image.size}."
        raise EvaluationError(message=msg)
    return np.array(image.convert("RGB"), dtype=np.uint8)


def mean_squared_error(cover: UInt8Array, stego: UInt8Array) -> float:
    """
    Mean squared error between original and stego RGB pixel values.

    Args:
        cover: Original image array, shape (H, W) or (H, W, 3).
        stego: Stego image array, identical shape to `cover`.

    Returns:
        MSE in ``[0.0, MAX_PIXEL_VALUE**2]``; 0.0 iff the arrays are equal.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    cover_arr = _validate_array(cover, "cover")
    stego_arr = _validate_array(stego, "stego")
    if cover_arr.shape != stego_arr.shape:
        msg = f"MSE requires identical shapes, got {cover_arr.shape} vs {stego_arr.shape}."
        raise ValueError(msg)
    diff = cover_arr.astype(np.float64) - stego_arr.astype(np.float64)
    return float(np.mean(diff * diff))


def peak_signal_to_noise_ratio(
    cover: UInt8Array,
    stego: UInt8Array,
    *,
    mse: float | None = None,
    max_value: float = MAX_PIXEL_VALUE,
) -> float:
    """
    Peak signal-to-noise ratio, mathematically derived from MSE.

    PSNR = 10 * log10(max_value**2 / MSE). Returns ``math.inf`` when MSE is
    zero (identical images) — the correct, exact answer, not an overflow.

    Args:
        cover: Original image array, shape (H, W) or (H, W, 3).
        stego: Stego image array, identical shape to `cover`.
        mse: Precomputed MSE to avoid recomputation. When omitted, MSE is
            computed from `cover`/`stego`.
        max_value: The peak signal value; 255 for 8-bit images.

    Returns:
        PSNR in dB (higher is better), or ``math.inf`` for identical images.

    Raises:
        ValueError: If the arrays differ in shape, or `mse`/`max_value` are
            negative (a non-positive `max_value` has no log-domain meaning).
    """
    mse_value = mean_squared_error(cover, stego) if mse is None else mse
    if mse_value < 0.0:
        msg = f"mse must be non-negative, got {mse_value}."
        raise ValueError(msg)
    if max_value <= 0.0:
        msg = f"max_value must be positive, got {max_value}."
        raise ValueError(msg)
    if mse_value == 0.0:
        return math.inf
    return float(10.0 * math.log10((max_value * max_value) / mse_value))


def structural_similarity_map(
    cover: UInt8Array,
    stego: UInt8Array,
    *,
    window_size: int = _SSIM_WINDOW_SIZE,
    sigma: float = _SSIM_SIGMA,
    k1: float = _SSIM_K1,
    k2: float = _SSIM_K2,
    data_range: float = MAX_PIXEL_VALUE,
) -> FloatArray:
    """
    Per-pixel structural similarity map (Wang et al. 2004) between two images.

    Computes luminance/contrast/structure statistics over Gaussian windows
    with reflect-padded convolution, returning one SSIM value per pixel (per
    channel, for RGB input). 1.0 where the images are locally identical;
    values below zero indicate anticorrelated structure.

    Args:
        cover: Original image array, shape (H, W) or (H, W, 3).
        stego: Stego image array, identical shape to `cover`.
        window_size: Odd Gaussian window size; clamped to the smaller image
            axis when the image is smaller than the requested window.
        sigma: Gaussian window standard deviation.
        k1, k2: Stabilization constants for the luminance/contrast terms.
        data_range: Peak signal value used to scale C1/C2; 255 for 8-bit.

    Returns:
        SSIM map of shape (H, W) for grayscale input, or (H, W, 3) for RGB
        input, in ``[-1.0, 1.0]``.

    Raises:
        ValueError: If the arrays differ in shape, `window_size` is even or
            non-positive, or `data_range` is non-positive.
    """
    cover_arr = _validate_array(cover, "cover")
    stego_arr = _validate_array(stego, "stego")
    if cover_arr.shape != stego_arr.shape:
        msg = f"SSIM requires identical shapes, got {cover_arr.shape} vs {stego_arr.shape}."
        raise ValueError(msg)
    if window_size < 1 or window_size % 2 == 0:
        msg = f"window_size must be a positive odd integer, got {window_size}."
        raise ValueError(msg)
    if data_range <= 0.0:
        msg = f"data_range must be positive, got {data_range}."
        raise ValueError(msg)

    effective = effective_window(window_size, cover_arr.shape)
    kernel = gaussian_kernel(effective, sigma)

    x = cover_arr.astype(np.float64)
    y = stego_arr.astype(np.float64)

    mu_x = convolve2d(x, kernel)
    mu_y = convolve2d(y, kernel)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = convolve2d(x * x, kernel) - mu_x2
    sigma_y2 = convolve2d(y * y, kernel) - mu_y2
    sigma_xy = convolve2d(x * y, kernel) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return numerator / denominator


def structural_similarity(
    cover: UInt8Array,
    stego: UInt8Array,
    *,
    window_size: int = _SSIM_WINDOW_SIZE,
    sigma: float = _SSIM_SIGMA,
    k1: float = _SSIM_K1,
    k2: float = _SSIM_K2,
    data_range: float = MAX_PIXEL_VALUE,
) -> float:
    """
    Mean structural similarity (Wang et al. 2004) between cover and stego.

    Convenience wrapper over structural_similarity_map: returns the mean SSIM
    over every pixel (and every channel, for RGB input). 1.0 iff the images
    are identical; values below zero indicate anticorrelated structure.

    Args:
        cover: Original image array, shape (H, W) or (H, W, 3).
        stego: Stego image array, identical shape to `cover`.
        window_size: Odd Gaussian window size; clamped to the smaller image
            axis when the image is smaller than the requested window.
        sigma: Gaussian window standard deviation.
        k1, k2: Stabilization constants for the luminance/contrast terms.
        data_range: Peak signal value used to scale C1/C2; 255 for 8-bit.

    Returns:
        Mean SSIM in ``[-1.0, 1.0]``.

    Raises:
        ValueError: If the arrays differ in shape, `window_size` is even or
            non-positive, or `data_range` is non-positive.
    """
    ssim_map = structural_similarity_map(
        cover,
        stego,
        window_size=window_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
        data_range=data_range,
    )
    return float(np.mean(ssim_map))


def image_quality(cover: Image.Image, stego: Image.Image) -> ImageQualityMetrics:
    """
    Compute MSE/PSNR/SSIM between two PIL images (RGB-normalized).

    Convenience wrapper over the array metrics; the pair is converted once
    through ``image_to_rgb_array`` so all three metrics share the same
    representation.

    Raises:
        EvaluationError: If either input is not a PIL image or is empty.
    """
    cover_arr = image_to_rgb_array(cover)
    stego_arr = image_to_rgb_array(stego)
    mse = mean_squared_error(cover_arr, stego_arr)
    psnr = peak_signal_to_noise_ratio(cover_arr, stego_arr, mse=mse)
    ssim = structural_similarity(cover_arr, stego_arr)
    return ImageQualityMetrics(mse=mse, psnr=psnr, ssim=ssim)


# --- Internal helpers ---------------------------------------------------------


def _validate_array(arr: object, name: str) -> UInt8Array:
    """Validate that `arr` is a non-empty 2D/3D numpy array."""
    if not isinstance(arr, np.ndarray):
        msg = f"{name} must be a numpy array, got {type(arr).__name__}."
        raise ValueError(msg)
    if arr.ndim not in (2, 3):
        msg = f"{name} must be 2D or 3D (H, W, C), got {arr.ndim} dimensions."
        raise ValueError(msg)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        msg = f"{name} must have non-empty spatial dimensions, got {arr.shape}."
        raise ValueError(msg)
    return arr


# --- Reflect-padded Gaussian window primitives (public) ----------------------
#
# These are the shared window building blocks behind structural_similarity_map
# and the Phase 2.8 label generator: an odd Gaussian window clamped to the
# image, and reflect-padded convolution. They are public so ai.labels derives
# its single-pixel marginal-cost label from the exact same window the metric
# layer uses — the label signal is definitionally consistent with SSIM.


def effective_window(window_size: int, shape: tuple[int, ...]) -> int:
    """
    Clamp `window_size` so it never exceeds the smaller image axis.

    Reflect padding needs at least one interior pixel, so a 1x1 image
    degrades to a 1x1 window (and preserves the odd-size invariant).
    """
    smallest = min(shape[0], shape[1])
    if smallest >= window_size:
        return window_size
    if smallest % 2 == 1:
        return smallest
    return smallest - 1 if smallest - 1 >= 1 else 1


def gaussian_kernel(size: int, sigma: float) -> FloatArray:
    """Normalized Gaussian kernel of shape (size, size); sums to 1.0."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = np.outer(g, g)
    return kernel / float(kernel.sum())


def convolve2d(arr: FloatArray, kernel: FloatArray) -> FloatArray:
    """
    2D convolution of `kernel` over `arr` with reflect padding, same size out.

    `arr` may be (H, W) or (H, W, C); the kernel is applied per channel.
    Reflect (mirror) padding means the output is defined at every pixel,
    including image borders — the same edge policy core.adaptive uses.
    """
    kh = int(kernel.shape[0])
    kw = int(kernel.shape[1])
    pad_h = kh // 2
    pad_w = kw // 2
    pad_width: tuple[tuple[int, int], ...] = ((pad_h, pad_h), (pad_w, pad_w))
    pad_width += ((0, 0),) * (arr.ndim - 2)
    padded = np.pad(arr, pad_width, mode="reflect")
    result = np.zeros_like(arr, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            coeff = float(kernel[i, j])
            result += coeff * padded[i : i + arr.shape[0], j : j + arr.shape[1]]
    return result
