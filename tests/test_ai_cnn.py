"""
Phase 2.8.3 CNN baseline tests.

Covers ai.cnn, fully offline:
- the Z input domain (``image & 0xFE``) and its input validation
- conv2d against a naive reference and its backward against finite differences
  (both weights and input), plus ReLU/sigmoid backward
- the full network: output shape/range for any HxW (incl. odd and 1x1),
  deterministic He init, predict == forward, and an end-to-end gradient check
  of ``backward`` against finite differences of the MSE loss
- invalid inputs raise ValueError

No ML framework and no network are involved; all inputs are in-memory.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ai.cnn import (
    SuitabilityCNN,
    conv2d,
    conv2d_backward,
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    z_domain_array,
)
from ai.training import mean_squared_error


def _mse_grad(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of MSE (mean over all elements) w.r.t. the prediction."""
    return 2.0 * (prediction - target) / float(prediction.size)


# --- Helpers ------------------------------------------------------------------


def _naive_conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Naive (k, k) SAME zero-padded convolution reference for conv2d."""
    batch, height, width, _ = x.shape
    kernel = weight.shape[0]
    channels_out = weight.shape[3]
    pad = kernel // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    out = np.zeros((batch, height, width, channels_out), dtype=np.float64)
    for i in range(kernel):
        for j in range(kernel):
            out += np.einsum(
                "bhwc,cn->bhwn",
                padded[:, i : i + height, j : j + width, :],
                weight[i, j],
                optimize=False,
            )
    return out + bias


def _numerical_gradient(loss, param_flat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of `loss` w.r.t. the flat params."""
    grad = np.zeros_like(param_flat)
    for i in range(param_flat.size):
        original = param_flat[i]
        param_flat[i] = original + eps
        high = loss(param_flat)
        param_flat[i] = original - eps
        low = loss(param_flat)
        param_flat[i] = original
        grad[i] = (high - low) / (2.0 * eps)
    return grad


def _flat(model: SuitabilityCNN) -> np.ndarray:
    """All parameters as one flat array: per layer, weight then bias."""
    return np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in model.parameters()])


def _set_flat(model: SuitabilityCNN, flat: np.ndarray) -> None:
    """Overwrite the model's parameters in place from a flat array."""
    offset = 0
    for w, b in model.parameters():
        w.ravel()[:] = flat[offset : offset + w.size]
        offset += w.size
        b.ravel()[:] = flat[offset : offset + b.size]
        offset += b.size


# --- Z domain -----------------------------------------------------------------


class TestZDomain:
    def test_clears_every_lsb(self) -> None:
        arr = np.array([[[0, 1, 2], [253, 254, 255]]], dtype=np.uint8)
        z = z_domain_array(arr)
        assert (z % 2 == 0).all()
        assert np.array_equal(z, arr & np.uint8(0xFE))

    def test_accepts_pil_and_rgb_normalizes(self) -> None:
        rgba = Image.new("RGBA", (4, 3), color=(255, 128, 0, 99))
        z = z_domain_array(rgba)
        assert z.shape == (3, 4, 3)
        assert np.array_equal(z, np.full((3, 4, 3), (254, 128, 0), dtype=np.uint8))

    def test_rejects_non_image_inputs(self) -> None:
        with pytest.raises(ValueError, match="PIL image or"):
            z_domain_array("not-an-image")  # type: ignore[arg-type]

    def test_rejects_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            z_domain_array(np.zeros((4, 4), dtype=np.uint8))
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            z_domain_array(np.zeros((4, 4, 1), dtype=np.uint8))

    def test_rejects_empty_spatial_dims(self) -> None:
        with pytest.raises(ValueError, match="positive width and height"):
            z_domain_array(np.zeros((0, 4, 3), dtype=np.uint8))


# --- Convolution primitives ----------------------------------------------------


class TestConv2d:
    def test_matches_naive_reference(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 5, 6, 3))
        weight = rng.normal(size=(3, 3, 3, 4)) * 0.2
        bias = rng.normal(size=(4,)) * 0.1
        assert np.allclose(conv2d(x, weight, bias), _naive_conv2d(x, weight, bias), atol=1e-12)

    def test_keeps_spatial_size(self) -> None:
        x = np.zeros((2, 7, 9, 3))
        weight = np.zeros((5, 5, 3, 4))
        bias = np.zeros((4,))
        assert conv2d(x, weight, bias).shape == (2, 7, 9, 4)

    def test_backward_matches_finite_differences_on_weights(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 4, 4, 3))
        weight = rng.normal(size=(3, 3, 3, 4)) * 0.2
        bias = rng.normal(size=(4,)) * 0.1
        target = rng.normal(size=(2, 4, 4, 4))

        def loss(flat: np.ndarray) -> float:
            w = flat[: weight.size].reshape(weight.shape)
            b = flat[weight.size :]
            return float(((conv2d(x, w, b) - target) ** 2).mean())

        flat = np.concatenate([weight.ravel(), bias.ravel()])
        numeric = _numerical_gradient(loss, flat.copy())
        dout = 2.0 * (conv2d(x, weight, bias) - target) / target.size
        _, grad_w, grad_b = conv2d_backward(x, weight, dout)
        analytic = np.concatenate([grad_w.ravel(), grad_b.ravel()])
        assert np.allclose(analytic, numeric, atol=1e-6)

    def test_backward_matches_finite_differences_on_input(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(2, 4, 4, 3))
        weight = rng.normal(size=(3, 3, 3, 4)) * 0.2
        bias = rng.normal(size=(4,)) * 0.1
        target = rng.normal(size=(2, 4, 4, 4))

        def loss(x_flat: np.ndarray) -> float:
            out = conv2d(x_flat.reshape(x.shape), weight, bias)
            return float(((out - target) ** 2).mean())

        numeric = _numerical_gradient(loss, x.ravel().copy())
        dout = 2.0 * (conv2d(x, weight, bias) - target) / target.size
        grad_x, _, _ = conv2d_backward(x, weight, dout)
        assert np.allclose(grad_x.ravel(), numeric, atol=1e-6)

    def test_rejects_even_kernel(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            conv2d(np.zeros((1, 4, 4, 3)), np.zeros((2, 2, 3, 4)), np.zeros((4,)))


class TestActivations:
    def test_relu_backward_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(3, 5))
        x[np.abs(x) < 0.05] += 0.5  # keep off the kink for a smooth numeric check
        g = rng.normal(size=(3, 5))

        def loss(x_flat: np.ndarray) -> float:
            return float((relu(x_flat.reshape(x.shape)) * g).sum())

        numeric = _numerical_gradient(loss, x.ravel().copy())
        assert np.allclose(relu_backward(x, g).ravel(), numeric, atol=1e-6)

    def test_sigmoid_backward_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(3, 5))
        g = rng.normal(size=(3, 5))

        def loss(x_flat: np.ndarray) -> float:
            return float((sigmoid(x_flat.reshape(x.shape)) * g).sum())

        numeric = _numerical_gradient(loss, x.ravel().copy())
        assert np.allclose(sigmoid_backward(x, g).ravel(), numeric, atol=1e-6)

    def test_sigmoid_outputs_stay_in_unit_interval(self) -> None:
        out = sigmoid(np.array([-100.0, 0.0, 100.0]))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)


# --- The network ---------------------------------------------------------------


class TestSuitabilityCNN:
    def test_forward_output_shape_and_range(self) -> None:
        model = SuitabilityCNN(seed=0)
        x = np.random.default_rng(0).random((3, 16, 16, 3), dtype=np.float64)
        pred = model.forward(x)
        assert pred.shape == (3, 16, 16)
        assert pred.min() >= 0.0 and pred.max() <= 1.0

    def test_forward_requires_rgb_batch(self) -> None:
        model = SuitabilityCNN(seed=0)
        with pytest.raises(ValueError, match=r"\(B, H, W, 3\)"):
            model.forward(np.zeros((1, 8, 8, 4)))
        with pytest.raises(ValueError, match=r"\(B, H, W, 3\)"):
            model.forward(np.zeros((8, 8, 3)))

    def test_forward_handles_odd_and_tiny_sizes(self) -> None:
        model = SuitabilityCNN(seed=0)
        for size in ((7, 9), (1, 1), (48, 32)):
            x = np.zeros((1, *size, 3))
            assert model.forward(x).shape == (1, *size)

    def test_init_is_deterministic_and_seed_dependent(self) -> None:
        first = SuitabilityCNN(seed=7)
        second = SuitabilityCNN(seed=7)
        different = SuitabilityCNN(seed=8)
        assert np.allclose(_flat(first), _flat(second))
        assert not np.allclose(_flat(first), _flat(different))

    def test_depth_and_parameter_count(self) -> None:
        model = SuitabilityCNN(seed=0)
        assert model.depth == 5
        assert len(model.parameters()) == 5
        total = sum(w.size + b.size for w, b in model.parameters())
        assert 15000 < total < 20000  # small baseline, fixed architecture

    def test_predict_matches_forward(self) -> None:
        rng = np.random.default_rng(5)
        model = SuitabilityCNN(seed=1)
        z = rng.integers(0, 256, size=(12, 10, 3), dtype=np.uint8) & np.uint8(0xFE)
        x = np.asarray(z, dtype=np.float64)[None, ...] / 255.0
        assert np.allclose(model.predict(z), model.forward(x)[0], atol=1e-12)
        assert np.allclose(model.predict(z), model.predict(Image.fromarray(z)), atol=1e-12)

    def test_predict_accepts_any_pil_mode(self) -> None:
        model = SuitabilityCNN(seed=1)
        gray = Image.new("L", (9, 8), color=200)
        pred = model.predict(gray)
        assert pred.shape == (8, 9)

    def test_backward_matches_finite_differences_end_to_end(self) -> None:
        rng = np.random.default_rng(6)
        model = SuitabilityCNN(seed=2)
        x = rng.normal(size=(1, 8, 8, 3))
        target = rng.random(size=(1, 8, 8))
        flat = _flat(model).copy()

        pred, cache = model.forward_with_cache(x)
        grads = model.backward(cache, _mse_grad(pred, target))
        analytic = _flat_of_grads(grads)

        def loss_at(index: int, delta: float) -> float:
            perturbed = flat.copy()
            perturbed[index] += delta
            _set_flat(model, perturbed)
            p, _ = model.forward_with_cache(x)
            return mean_squared_error(p, target)

        eps = 1e-4
        indices = [i for i in (0, 1, 5, 40, 300, 1500, 7000, 12000, flat.size - 1) if i < flat.size]
        for index in indices:
            numeric = (loss_at(index, eps) - loss_at(index, -eps)) / (2.0 * eps)
            err = abs(analytic[index] - numeric)
            assert err <= 1e-2 * max(1.0, abs(numeric)), (index, err, analytic[index], numeric)


def _flat_of_grads(grads: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Flatten backward gradients into the same ordering as ``_flat``."""
    return np.concatenate([np.concatenate([dw.ravel(), db.ravel()]) for dw, db in grads])
