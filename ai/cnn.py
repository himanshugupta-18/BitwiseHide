"""
Suitability CNN baseline (Phase 2.8.3, training-time only).

A small, fully-convolutional, framework-agnostic (pure numpy) network that maps
an RGB image in the Z domain — every channel LSB cleared, ``Z = image & 0xFE`` —
to the per-pixel embedding-suitability label produced by
``ai.labels.suitability_label_map``. It is the Phase 2.8.3 baseline the later
phases build the trained artifact on: deliberately small and focused, not the
final production model.

Design
------
- Input domain: Z(image). The model never sees LSB values, matching the
  deterministic analysis domain the Phase 2.6 embedder and the Phase 2.8.2
  label both operate on, so predictions are defined on the same domain used at
  inference time.
- Constant resolution: every conv layer uses SAME zero padding, so the output
  map keeps the input HxW and the network accepts ANY image size. This matters
  because BSDS500 images have odd dimensions (e.g. 321x481); a pooling +
  upsampling design would force cropping to a fixed size.
- Receptive field: the five stacked convolutions (5x5, 3x3, 3x3, 3x3, 1x1)
  reach exactly an 11x11 field — the effective window size behind the label
  (``evaluation._SSIM_WINDOW_SIZE``) — so every prediction can in principle see
  the same neighborhood the label uses.
- Deterministic: weights are He-initialized from ``np.random.default_rng(seed)``
  and every op is single-threaded numpy. The same seed yields the same network,
  and the same network + same batch sequence yields the same gradients.
- Gradient: ``backward`` returns exact gradients of the loss w.r.t. the model
  output (MSE by convention), verified against finite differences in the tests.

Design constraints:
- Framework-agnostic: no torch or other ML framework is imported, mirroring
  ``ai.labels`` and ``app.core.evaluation``.
- Invalid input (a non-array/non-PIL value, or an array that is not
  ``(H, W, 3)``) raises ValueError — the caller is misconfiguring the call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from app.core.evaluation import FloatArray, UInt8Array

#: Byte mask that clears every channel LSB (the Z(image) analysis domain).
_LSB_CLEAR = 0xFE

#: Architecture: (input channels, output channels, kernel size) per conv layer.
_ARCHITECTURE: tuple[tuple[int, int, int], ...] = (
    (3, 16, 5),
    (16, 16, 3),
    (16, 32, 3),
    (32, 32, 3),
    (32, 1, 1),
)


def z_domain_array(image: Image.Image | UInt8Array) -> UInt8Array:
    """Return the RGB array of `image` with every channel LSB cleared (Z domain).

    `image` may be a PIL image (any mode; converted to RGB) or a ``(H, W, 3)``
    uint8 numpy array. ``Z = image & 0xFE`` is exactly the analysis domain the
    Phase 2.6 embedder operates on and the input domain this baseline is
    trained on, so the model's predictions are defined on the same domain used
    at inference time.

    Raises:
        ValueError: If `image` is neither a PIL image nor a ``(H, W, 3)`` uint8
            array, or has zero width/height.
    """
    if isinstance(image, np.ndarray):
        arr = image
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        msg = f"Expected a PIL image or (H, W, 3) uint8 array, got {type(image).__name__}."
        raise ValueError(msg)
    if arr.ndim != 3 or arr.shape[2] != 3:
        msg = f"Expected a (H, W, 3) array, got shape {arr.shape}."
        raise ValueError(msg)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        msg = f"Image must have positive width and height, got {arr.shape}."
        raise ValueError(msg)
    return (arr & np.uint8(_LSB_CLEAR)).astype(np.uint8)


def conv2d(
    x: FloatArray,
    weight: FloatArray,
    bias: FloatArray,
) -> FloatArray:
    """2D SAME zero-padded convolution over the (spatial, channel) axes.

    Args:
        x: ``(B, H, W, C_in)`` float64 batch.
        weight: ``(k, k, C_in, C_out)`` float64 kernel, `k` odd.
        bias: ``(C_out,)`` float64 bias.

    Returns:
        A ``(B, H, W, C_out)`` feature map, same spatial size as `x`.

    Raises:
        ValueError: If `weight` is not an odd kernel, or the shapes of `x`,
            `weight`, and `bias` are inconsistent.
    """
    kernel = int(weight.shape[0])
    if kernel % 2 == 0 or kernel < 1:
        msg = f"conv2d requires an odd positive kernel, got {kernel}."
        raise ValueError(msg)
    if int(weight.shape[2]) != int(x.shape[3]):
        msg = f"Conv input channels {x.shape[3]} do not match kernel channels {weight.shape[2]}."
        raise ValueError(msg)
    if int(weight.shape[3]) != int(bias.shape[0]):
        msg = f"Conv output channels {weight.shape[3]} do not match bias size {bias.shape[0]}."
        raise ValueError(msg)
    pad = kernel // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    patches = _im2col(padded, int(x.shape[1]), int(x.shape[2]), kernel)
    w_mat: FloatArray = np.asarray(weight.reshape(-1, int(weight.shape[3])), dtype=np.float64)
    return np.einsum("bhwn,nc->bhwc", patches, w_mat, optimize=False) + bias


def conv2d_backward(
    x: FloatArray,
    weight: FloatArray,
    grad_output: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Gradients of a SAME conv pass w.r.t. its input and parameters.

    Args:
        x: The forward-pass input ``(B, H, W, C_in)``.
        weight: The kernel ``(k, k, C_in, C_out)`` used in the forward pass.
        grad_output: Upstream gradient ``(B, H, W, C_out)``.

    Returns:
        ``(grad_x, grad_weight, grad_bias)`` with the shapes of `x`, `weight`,
        and the bias vector respectively.
    """
    kernel = int(weight.shape[0])
    pad = kernel // 2
    batch, height, width = x.shape[0], int(x.shape[1]), int(x.shape[2])
    channels_in = int(x.shape[3])
    channels_out = int(weight.shape[3])

    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    patches = _im2col(padded, height, width, kernel)

    w_mat: FloatArray = np.asarray(weight.reshape(-1, channels_out), dtype=np.float64)
    grad_weight: FloatArray = np.asarray(
        np.einsum("bhwn,bhwc->nc", patches, grad_output, optimize=False).reshape(
            kernel, kernel, channels_in, channels_out
        ),
        dtype=np.float64,
    )
    grad_bias: FloatArray = np.asarray(grad_output.sum(axis=(0, 1, 2)), dtype=np.float64)

    g_patches: FloatArray = np.asarray(
        np.einsum("bhwc,nc->bhwn", grad_output, w_mat, optimize=False), dtype=np.float64
    )
    grad_padded: FloatArray = np.zeros(
        (batch, height + 2 * pad, width + 2 * pad, channels_in), dtype=np.float64
    )
    for i in range(kernel):
        for j in range(kernel):
            start_channel = (i * kernel + j) * channels_in
            block = g_patches[..., start_channel : start_channel + channels_in]
            grad_padded[:, i : i + height, j : j + width, :] += block
    grad_x: FloatArray = grad_padded[:, pad : pad + height, pad : pad + width, :]
    return grad_x, grad_weight, grad_bias


def relu(x: FloatArray) -> FloatArray:
    """Rectified linear unit, elementwise."""
    return np.maximum(x, 0.0)


def relu_backward(x: FloatArray, grad_output: FloatArray) -> FloatArray:
    """Gradient of ReLU: pass `grad_output` through where `x` > 0, zero else."""
    return np.where(x > 0.0, grad_output, 0.0)


def sigmoid(x: FloatArray) -> FloatArray:
    """Elementwise logistic sigmoid, clipped for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def sigmoid_backward(x: FloatArray, grad_output: FloatArray) -> FloatArray:
    """Gradient of sigmoid at the pre-activation `x`, times `grad_output`."""
    y = sigmoid(x)
    return grad_output * y * (1.0 - y)


@dataclass(frozen=True)
class ForwardCache:
    """Intermediate activations of one forward pass, for the training backward.

    Attributes:
        inputs: ``inputs[i]`` is the input to conv layer `i` (post-activation
            of the previous layer; ``inputs[0]`` is the model input).
        pre: ``pre[i]`` is the pre-activation (before ReLU/sigmoid) of layer `i`.
    """

    inputs: tuple[FloatArray, ...]
    pre: tuple[FloatArray, ...]


class SuitabilityCNN:
    """Small constant-resolution CNN: Z-domain RGB image -> HxW suitability map.

    The network maps a normalized Z-domain batch ``(B, H, W, 3)`` in [0, 1] to
    per-pixel suitability predictions ``(B, H, W)`` in [0, 1], supervised by
    ``ai.labels.suitability_label_map``. Construction is deterministic: the
    weights are He-initialized from ``np.random.default_rng(seed)``, so the
    same seed reproduces the same network (and the same training run).
    """

    def __init__(self, *, seed: int = 0) -> None:
        """Initialize the network with deterministic He-initialized weights."""
        self.seed = seed
        rng = np.random.default_rng(seed)
        self._weights: list[FloatArray] = []
        self._biases: list[FloatArray] = []
        for channels_in, channels_out, kernel in _ARCHITECTURE:
            fan_in = kernel * kernel * channels_in
            std = math.sqrt(2.0 / fan_in)  # He init for ReLU activations.
            weight = rng.normal(0.0, std, size=(kernel, kernel, channels_in, channels_out))
            self._weights.append(np.asarray(weight, dtype=np.float64))
            self._biases.append(np.zeros(channels_out, dtype=np.float64))

    @property
    def depth(self) -> int:
        """Number of conv layers in the network."""
        return len(self._weights)

    def parameters(self) -> list[tuple[FloatArray, FloatArray]]:
        """``(weight, bias)`` pairs, one per conv layer, in forward order."""
        return list(zip(self._weights, self._biases, strict=True))

    def forward(self, x: FloatArray) -> FloatArray:
        """Predict ``(B, H, W)`` suitability maps in [0, 1] for a ``(B, H, W, 3)`` batch.

        `x` is the normalized Z-domain input (float64 in [0, 1]). No activation
        cache is retained — use ``forward_with_cache`` for training.

        Raises:
            ValueError: If `x` is not a ``(B, H, W, 3)`` batch.
        """
        if x.ndim != 4 or x.shape[3] != 3:
            msg = f"SuitabilityCNN expects a (B, H, W, 3) batch, got shape {x.shape}."
            raise ValueError(msg)
        output, _ = self.forward_with_cache(x)
        return output

    def forward_with_cache(self, x: FloatArray) -> tuple[FloatArray, ForwardCache]:
        """Forward pass returning the ``(B, H, W)`` prediction and a training cache.

        The cache records per-layer inputs and pre-activations so ``backward``
        reproduces the activations exactly without recomputation.
        """
        inputs: list[FloatArray] = [x]
        pre: list[FloatArray] = []
        h = x
        for index, (weight, bias) in enumerate(self.parameters()):
            h = conv2d(h, weight, bias)
            pre.append(h)
            h = relu(h) if index < self.depth - 1 else sigmoid(h)
            inputs.append(h)
        cache = ForwardCache(inputs=tuple(inputs), pre=tuple(pre))
        return h[..., 0], cache

    def backward(
        self,
        cache: ForwardCache,
        grad_output: FloatArray,
    ) -> list[tuple[FloatArray, FloatArray]]:
        """Backpropagate the gradient of the loss w.r.t. the ``(B, H, W)`` output.

        Args:
            cache: From ``forward_with_cache`` for the same batch.
            grad_output: ``(B, H, W)`` upstream gradient, e.g. the MSE gradient
                returned by ``ai.training._mse_gradient``.

        Returns:
            Per-layer ``(grad_weight, grad_bias)`` pairs in forward order.
        """
        grad = grad_output[..., None]
        grads: list[tuple[FloatArray, FloatArray]] = []
        for index in range(self.depth - 1, -1, -1):
            grad = (
                relu_backward(cache.pre[index], grad)
                if index < self.depth - 1
                else sigmoid_backward(cache.pre[index], grad)
            )
            grad_x, grad_weight, grad_bias = conv2d_backward(
                cache.inputs[index],
                self._weights[index],
                grad,
            )
            grads.append((grad_weight, grad_bias))
            grad = grad_x
        grads.reverse()
        return grads

    def predict(self, image: Image.Image | UInt8Array) -> FloatArray:
        """Predict a single ``(H, W)`` suitability map for a Z-domain image.

        `image` may be a PIL image (any mode) or a ``(H, W, 3)`` uint8 array;
        the Z-domain is derived with ``z_domain_array`` and normalized to
        [0, 1] before the forward pass. Returns the ``(H, W)`` prediction.

        Raises:
            ValueError: If `image` is not a PIL image or a ``(H, W, 3)`` array.
        """
        z = z_domain_array(image)
        x = np.asarray(z, dtype=np.float64)[None, ...] / 255.0
        output, _ = self.forward_with_cache(x)
        return output[0]


# --- Internal helpers ---------------------------------------------------------


def _im2col(padded: FloatArray, height: int, width: int, kernel: int) -> FloatArray:
    """Extract sliding kxk channel blocks of `padded` as im2col columns.

    `padded` is ``(B, H+2p, W+2p, C)``; the result is ``(B, H, W, k*k*C)``
    where column index ``(i*k + j) * C + c`` is channel `c` at kernel offset
    ``(i, j)`` — the exact ordering ``weight.reshape(-1, C_out)`` assumes.
    """
    window = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel), axis=(1, 2))
    # sliding_window_view orders its axes (B, H, W, C, i, j); transpose the
    # kernel axes before the channel axis so flattening yields (i, j) slowest.
    reordered = np.transpose(window, (0, 1, 2, 4, 5, 3))
    batch = int(padded.shape[0])
    channels = int(padded.shape[3])
    return np.asarray(
        reordered.reshape(batch, height, width, kernel * kernel * channels),
        dtype=np.float64,
    )
