"""
Embedding-suitability label generation (Phase 2.8, training-time only).

Computes a per-pixel suitability label map for an image: how well each pixel's
neighborhood tolerates the embedder's atomic write. The label at pixel p is the
local structural similarity (SSIM, Wang et al. 2004) between the LSB-cleared
analysis domain and that domain with a single +1 perturbation at p:

    Z(image) = every RGB channel LSB cleared (byte & 0xFE)
    Z + E_p  = Z with all three RGB channels at pixel p incremented by 1

The suitability of p is the mean-over-RGB-channels SSIM its neighborhood keeps
under that single-pixel +1 flip, clamped to [0, 1] (higher = more suitable).
Z(image) is exactly the analysis domain Phase 2.6's embedder and extractor
operate on, so the label signal matches the deterministic domain used at
inference time.

Why a single-pixel probe and not the earlier Z vs C' probe
----------------------------------------------------------
The original label compared Z(image) against C'(image) = image | 0x01, i.e. a
uniform +1 shift of every channel. A constant additive offset leaves SSIM's
contrast and structure terms exactly invariant (sigma_y = sigma_x and
sigma_xy = sigma_x**2), so that map collapsed to a brightness-only luminance
term — no content discrimination. A single-pixel +1 spike is different: it
injects local variance (w - w**2) into a flat window and a tiny covariance
defect w*(x0 - m) into a textured one, so the contrast term drops measurably in
smooth regions and stays near 1 in textured regions. The label is therefore
"inverse local variance" derived from the metric layer — the contrast-masking
intuition behind adaptive LSB.

Closed form (no per-pixel SSIM calls)
-------------------------------------
With m = k x Z (local weighted mean), v = k x (Z**2) - m**2 (local weighted
variance), x0 = Z(p), and w(p) the total kernel weight pixel p receives within
its own window under reflect padding:

    mu'      = m + w(p)
    sigma'^2 = v + 2*w(p)*(x0 - m) + w(p) - w(p)**2
    sigma_xy = v + w(p)*(x0 - m)

    suitability(p) = clamp_c( (2 m mu' + C1)(2 sigma_xy + C2)
                            / ((m**2 + mu'**2 + C1)(v + sigma'^2 + C2)) )

The reflect-padding detail matters: near a border, numpy reflect padding
mirrors the perturbed pixel into its own window, so the spike contributes
w(p) = k0 + sum of reflection weights rather than just the kernel center k0.
The closed form accounts for this, making it EXACTLY equal to the per-pixel
probe structural_similarity_map(Z, Z + E_p) at every pixel (verified to
machine precision). Two Gaussian convolutions produce the full map — cheaper
than a single full SSIM and H*W times cheaper than the naive per-pixel probe.

Design constraints:
- Deterministic: the map is a pure function of the input image pixels.
- Image-only: the function never receives or uses payload, plaintext,
  ciphertext, or passwords.
- Framework-agnostic: no torch or other ML framework is imported; the label
  uses numpy and the public reflect-padded window primitives from
  app.core.evaluation (gaussian_kernel / effective_window / convolve2d).
- Invalid input (non-PIL, or zero width/height) raises ValueError — the caller
  is misconfiguring the call, mirroring the weight-validation split elsewhere.
"""

from __future__ import annotations

import numpy as np
from app.core.evaluation import (
    _SSIM_K1,
    _SSIM_K2,
    _SSIM_SIGMA,
    _SSIM_WINDOW_SIZE,
    MAX_PIXEL_VALUE,
    FloatArray,
    UInt8Array,
    convolve2d,
    effective_window,
    gaussian_kernel,
)
from PIL import Image

#: Byte mask that clears every channel LSB (the Z(image) analysis domain).
_LSB_CLEAR = 0xFE


def suitability_label_map(image: Image.Image) -> FloatArray:
    """
    Compute the HxW embedding-suitability label map for `image`.

    The label at pixel (y, x) is the mean-over-RGB-channels SSIM that the
    pixel's neighborhood retains when all three of its channels are flipped by
    +1 on the LSB-cleared domain Z(image) (the atomic Phase 2.6 write),
    clamped to [0, 1]. Smooth regions whose structure a +-1 flip would visibly
    damage score low; pixels whose local structure hides the flip score high.

    Args:
        image: A PIL image (any mode; normalized to RGB internally). The image
            is never mutated.

    Returns:
        A float64 map of shape (H, W) with every value in [0.0, 1.0].

    Raises:
        ValueError: If `image` is not a PIL image or has zero width/height.
    """
    if not isinstance(image, Image.Image):
        msg = f"Label generation requires a PIL image, got {type(image).__name__}."
        raise ValueError(msg)
    if image.size[0] <= 0 or image.size[1] <= 0:
        msg = f"Image must have positive width and height, got {image.size}."
        raise ValueError(msg)

    rgb = image.convert("RGB")
    z = (np.asarray(rgb, dtype=np.uint8) & _LSB_CLEAR).astype(np.uint8)
    return _marginal_suitability(z)


def _marginal_suitability(z: UInt8Array) -> FloatArray:
    """
    Closed-form mean-over-channels SSIM of Z vs the single-pixel +1 probe.

    `z` must be the LSB-cleared analysis domain of shape (H, W, 3). Returns the
    HxW suitability map in [0, 1]. This is EXACTLY
    structural_similarity_map(z, z + E_p) averaged over channels, evaluated for
    every pixel in O(H*W) via the window-statistics closed form in the module
    docstring.
    """
    x = z.astype(np.float64)
    window = effective_window(_SSIM_WINDOW_SIZE, x.shape)
    kernel = gaussian_kernel(window, _SSIM_SIGMA)

    mean = convolve2d(x, kernel)
    variance = convolve2d(x * x, kernel) - mean * mean

    c1 = (_SSIM_K1 * MAX_PIXEL_VALUE) ** 2
    c2 = (_SSIM_K2 * MAX_PIXEL_VALUE) ** 2

    weight = _spike_weight(x.shape[0], x.shape[1], kernel)[..., None]

    mean_prime = mean + weight
    variance_prime = variance + 2.0 * weight * (x - mean) + weight - weight * weight
    covariance = variance + weight * (x - mean)

    numerator = (2.0 * mean * mean_prime + c1) * (2.0 * covariance + c2)
    denominator = (mean * mean + mean_prime * mean_prime + c1) * (variance + variance_prime + c2)
    ssim_map = numerator / denominator
    ssim_map = ssim_map.mean(axis=2)
    return np.clip(ssim_map, 0.0, 1.0)


def _spike_weight(height: int, width: int, kernel: FloatArray) -> FloatArray:
    """
    Per-pixel total kernel weight of a +1 spike within its own window.

    For interior pixels this is the kernel center weight k0. Within `pad`
    positions of a border, reflect padding mirrors the perturbed pixel into its
    own window (row y maps to padding row -y), so the spike appears at several
    window positions at once; this returns the sum of every such copy's kernel
    weight as an HxW array. The exact corners have no mirror image and equal
    the interior value.
    """
    pad = kernel.shape[0] // 2
    center = pad
    weight = np.zeros((height, width), dtype=np.float64)
    for dy in range(-pad, pad + 1):
        rows = [y for y in range(height) if _reflect_index(y + dy, height) == y]
        for dx in range(-pad, pad + 1):
            cols = [x for x in range(width) if _reflect_index(x + dx, width) == x]
            if rows and cols:
                weight[np.ix_(rows, cols)] += kernel[center + dy, center + dx]
    return weight


def _reflect_index(index: int, size: int) -> int:
    """numpy reflect-padding index: mirror negative and overflow indices."""
    if index < 0:
        return -index
    if index >= size:
        return 2 * (size - 1) - index
    return index
