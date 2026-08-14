"""
Training, validation, and evaluation for the suitability CNN baseline
(Phase 2.8.3, training-time only).

A deterministic, framework-agnostic (pure numpy) training loop: MSE regression
from the Z-domain input to the Phase 2.8.2 suitability label, optimized with
Adam (Kingma & Ba 2015). Reproducibility comes from three fixed points:

- the model is seeded at construction (``SuitabilityCNN(seed=...)``);
- the batch permutation is drawn from a fresh ``np.random.default_rng`` seeded
  from ``TrainConfig.seed`` per call;
- every update is a fixed sequence of single-threaded numpy operations.

So the same model + same config reproduces the exact same loss history, and the
same config with a different seed gives a different history.

Two entry points:
- ``train`` runs a full training loop, records per-epoch train MSE and (when a
  validation dataset is supplied) validation MSE, and leaves the model updated
  in place.
- ``evaluate`` measures MSE/MAE/count of a model over a whole split in fixed
  record order (shuffling-free), giving the deterministic per-split metrics the
  experiment layer will consume.

Design constraints:
- Framework-agnostic: no torch or other ML framework is imported.
- Invalid *arguments* (empty dataset, non-positive epochs, shape-mismatched
  metric inputs) raise ValueError; corrupt dataset contents surface as
  ``DatasetError`` from the dataloader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from app.core.evaluation import FloatArray

    from ai.cnn import SuitabilityCNN
    from ai.dataloader import SuitabilityDataset

#: Fixed batch size used by ``evaluate`` (irrelevant to training results).
_EVAL_BATCH_SIZE = 16


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one deterministic training run."""

    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if not (0.0 <= self.beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}.")
        if not (0.0 <= self.beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")


@dataclass(frozen=True)
class TrainingHistory:
    """Per-epoch losses of one training run.

    Attributes:
        train_loss: Per-epoch train MSE, one entry per epoch.
        val_loss: Per-epoch validation MSE (empty when no validation dataset).
        epochs: The number of epochs run.
    """

    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]
    epochs: int


@dataclass(frozen=True)
class SplitMetrics:
    """Regression metrics of a model over one dataset split."""

    mse: float
    mae: float
    count: int


def mean_squared_error(prediction: FloatArray, target: FloatArray) -> float:
    """Mean squared error between two same-shaped arrays.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    if prediction.shape != target.shape:
        msg = f"MSE requires matching shapes, got {prediction.shape} vs {target.shape}."
        raise ValueError(msg)
    diff = prediction - target
    return float((diff * diff).mean())


def mean_absolute_error(prediction: FloatArray, target: FloatArray) -> float:
    """Mean absolute error between two same-shaped arrays.

    Raises:
        ValueError: If the arrays differ in shape.
    """
    if prediction.shape != target.shape:
        msg = f"MAE requires matching shapes, got {prediction.shape} vs {target.shape}."
        raise ValueError(msg)
    return float(np.abs(prediction - target).mean())


class AdamOptimizer:
    """Adam (Kingma & Ba, 2015) with bias correction, in pure numpy.

    Holds one first/second moment accumulator per model tensor (weights and
    biases separately) and updates the model's arrays in place. Deterministic:
    each step is a fixed sequence of numpy operations on fixed state, so the
    same model plus the same gradient sequence yields the exact same updates.
    """

    def __init__(
        self,
        model: SuitabilityCNN,
        *,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialize Adam state (zeroed moments) for every model tensor."""
        self._model = model
        self._learning_rate = float(learning_rate)
        self._beta1 = float(beta1)
        self._beta2 = float(beta2)
        self._eps = float(eps)
        self._step_count = 0
        self._state: list[tuple[FloatArray, FloatArray, FloatArray]] = []
        for weight, bias in model.parameters():
            self._state.append((weight, np.zeros_like(weight), np.zeros_like(weight)))
            self._state.append((bias, np.zeros_like(bias), np.zeros_like(bias)))

    def step(self, grads: Sequence[tuple[FloatArray, FloatArray]]) -> None:
        """Apply one Adam update from per-layer ``(grad_weight, grad_bias)``.

        Args:
            grads: In forward layer order, as returned by
                ``SuitabilityCNN.backward``.
        """
        self._step_count += 1
        t = float(self._step_count)
        b1 = self._beta1
        b2 = self._beta2
        lr = self._learning_rate
        for (param, moment1, moment2), grad in zip(self._state, _flatten_grads(grads), strict=True):
            moment1 *= b1
            moment1 += (1.0 - b1) * grad
            moment2 *= b2
            moment2 += (1.0 - b2) * (grad * grad)
            m_hat = moment1 / (1.0 - b1**t)
            v_hat = moment2 / (1.0 - b2**t)
            param -= lr * m_hat / (np.sqrt(v_hat) + self._eps)


def train(
    model: SuitabilityCNN,
    train_dataset: SuitabilityDataset,
    *,
    config: TrainConfig = TrainConfig(),
    val_dataset: SuitabilityDataset | None = None,
    epoch_callback: Callable[[int, SuitabilityCNN], None] | None = None,
) -> TrainingHistory:
    """Train `model` on `train_dataset` and return per-epoch MSE history.

    `model` is updated in place; re-create it (same seed) for a fresh start.
    Each epoch shuffles the training records with a fresh seeded RNG, so a given
    ``config.seed`` always reproduces the same batch sequence and history. When
    `val_dataset` is given, validation MSE is recorded after each train pass.

    Args:
        model: The CNN to train (seeded at construction).
        train_dataset: One split's (Z-domain input, label) pairs.
        config: Hyperparameters, including the batch-shuffle seed.
        val_dataset: Optional held-out split for per-epoch validation.
        epoch_callback: Optional hook invoked after each epoch (post-validation,
            zero-based ``(epoch_index, model)``). Because it runs after the
            model is fully updated for that epoch, a snapshot taken inside it
            reproduces the state whose validation loss was just recorded — the
            mechanism Phase 2.8.4's best-by-validation selection relies on. The
            hook must not mutate the model.

    Returns:
        The per-epoch train/validation MSE history.

    Raises:
        ValueError: If `train_dataset` is empty or `config.epochs` < 1.
    """
    if config.epochs < 1:
        msg = f"epochs must be at least 1, got {config.epochs}."
        raise ValueError(msg)
    if len(train_dataset) == 0:
        msg = "Cannot train on an empty dataset."
        raise ValueError(msg)

    optimizer = AdamOptimizer(
        model,
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
    )
    rng = np.random.default_rng(config.seed)
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(config.epochs):
        sum_sq = 0.0
        count = 0
        for x, y in train_dataset.shuffled_batches(config.batch_size, rng):
            prediction, cache = model.forward_with_cache(x)
            grads = model.backward(cache, _mse_gradient(prediction, y))
            optimizer.step(grads)
            diff = prediction - y
            sum_sq += float((diff * diff).sum())
            count += int(y.size)
        train_losses.append(sum_sq / count)
        if val_dataset is not None:
            val_losses.append(evaluate(model, val_dataset).mse)
        if epoch_callback is not None:
            epoch_callback(epoch, model)
    return TrainingHistory(
        train_loss=tuple(train_losses),
        val_loss=tuple(val_losses),
        epochs=config.epochs,
    )


def evaluate(model: SuitabilityCNN, dataset: SuitabilityDataset) -> SplitMetrics:
    """Evaluate `model` over every sample of `dataset` in fixed record order.

    Batches are taken in the dataset's record order (no shuffling), so results
    are deterministic and independent of any RNG. Returns the pixel-wise MSE
    and MAE over all samples plus the number of images.

    Raises:
        ValueError: If `dataset` is empty.
    """
    if len(dataset) == 0:
        msg = "Cannot evaluate on an empty dataset."
        raise ValueError(msg)
    sum_sq = 0.0
    sum_abs = 0.0
    count = 0
    for start in range(0, len(dataset), _EVAL_BATCH_SIZE):
        indices = list(range(start, min(start + _EVAL_BATCH_SIZE, len(dataset))))
        x, y = dataset.make_batch(indices)
        prediction = model.forward(x)
        diff = prediction - y
        sum_sq += float((diff * diff).sum())
        sum_abs += float(np.abs(diff).sum())
        count += int(y.size)
    return SplitMetrics(mse=sum_sq / count, mae=sum_abs / count, count=len(dataset))


def snapshot_parameters(model: SuitabilityCNN) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Deep-copy `model`'s (weight, bias) pairs in forward order.

    The snapshot is fully independent of the live model, so the model can keep
    training while the snapshot stays fixed. Phase 2.8.4 uses this to retain the
    validation-best model state for checkpointing.
    """
    return tuple((weight.copy(), bias.copy()) for weight, bias in model.parameters())


def restore_parameters(
    model: SuitabilityCNN,
    snapshot: Sequence[tuple[FloatArray, FloatArray]],
) -> None:
    """Overwrite `model`'s parameters in place from a `snapshot_parameters` copy.

    The snapshot must have been produced for the same architecture (same layer
    shapes); lengths are validated by the ``strict`` zip.

    Raises:
        ValueError: If `snapshot` has a different number of (weight, bias) pairs
            than `model`.
    """
    parameters = model.parameters()
    if len(snapshot) != len(parameters):
        msg = f"snapshot has {len(snapshot)} layers, model has {len(parameters)}."
        raise ValueError(msg)
    for (weight, bias), (saved_weight, saved_bias) in zip(parameters, snapshot, strict=True):
        weight[:] = saved_weight
        bias[:] = saved_bias


# --- Internal helpers ---------------------------------------------------------


def _mse_gradient(prediction: FloatArray, target: FloatArray) -> FloatArray:
    """Gradient of MSE = mean over all elements of ``(prediction - target)**2``."""
    return 2.0 * (prediction - target) / float(prediction.size)


def _flatten_grads(grads: Sequence[tuple[FloatArray, FloatArray]]) -> list[FloatArray]:
    """Interleave per-layer (weight, bias) gradients into one tensor list.

    The order matches ``AdamOptimizer._state``: one entry per weight, then one
    per bias, in forward layer order.
    """
    tensors: list[FloatArray] = []
    for grad_weight, grad_bias in grads:
        tensors.append(grad_weight)
        tensors.append(grad_bias)
    return tensors
