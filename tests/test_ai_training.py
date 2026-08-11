"""
Phase 2.8.3 training/validation/evaluation tests.

Covers ai.training, fully offline:
- MSE/MAE helpers (known values, shape validation)
- one Adam step descends the MSE loss
- a full training run on a deterministic synthetic split: loss decreases,
  validation loss is recorded per epoch, history shape is correct
- reproducibility: same seed reproduces the exact same history; a different
  seed gives a different history
- evaluate() matches a manual computation, is deterministic, and reports the
  sample count
- invalid arguments (empty dataset, zero epochs) raise ValueError

Everything runs in memory / tmp_path; no framework, no network.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ai.cnn import SuitabilityCNN
from ai.dataloader import SuitabilityDataset
from ai.prepare_dataset import write_synthetic_dataset
from ai.split import Split, resolve_split
from ai.training import (
    AdamOptimizer,
    SplitMetrics,
    TrainConfig,
    TrainingHistory,
    evaluate,
    mean_absolute_error,
    mean_squared_error,
    train,
)


def _splits(tmp_path, *, per_split: int = 3, size: tuple[int, int] = (16, 16)) -> dict:
    """Resolve a deterministic synthetic dataset into per-split records."""
    dataset = write_synthetic_dataset(tmp_path / "ds", per_split=per_split, size=size, seed=0)
    return resolve_split(dataset)


def _datasets(tmp_path, *, per_split: int = 3) -> tuple[SuitabilityDataset, SuitabilityDataset]:
    splits = _splits(tmp_path, per_split=per_split)
    return SuitabilityDataset(splits[Split.TRAIN]), SuitabilityDataset(splits[Split.VAL])


class TestMetrics:
    def test_mse_known_value(self) -> None:
        pred = np.array([0.0, 1.0, 2.0, 3.0])
        target = np.array([0.0, 0.0, 0.0, 0.0])
        assert mean_squared_error(pred, target) == pytest.approx(3.5)

    def test_mae_known_value(self) -> None:
        pred = np.array([0.0, 1.0, -2.0])
        target = np.zeros(3)
        assert mean_absolute_error(pred, target) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shapes"):
            mean_squared_error(np.zeros((2, 2)), np.zeros((3, 3)))
        with pytest.raises(ValueError, match="shapes"):
            mean_absolute_error(np.zeros((2, 2)), np.zeros((3, 3)))


class TestAdamStep:
    def test_one_step_reduces_loss(self) -> None:
        model = SuitabilityCNN(seed=0)
        x = np.random.default_rng(0).random((2, 8, 8, 3))
        y = np.random.default_rng(1).random((2, 8, 8))
        pred, cache = model.forward_with_cache(x)
        before = mean_squared_error(pred, y)
        grads = model.backward(cache, 2.0 * (pred - y) / float(pred.size))
        AdamOptimizer(model, learning_rate=1e-2).step(grads)
        after = mean_squared_error(model.forward(x), y)
        assert after < before


class TestTrain:
    def test_loss_decreases_on_synthetic_split(self, tmp_path) -> None:
        train_ds, val_ds = _datasets(tmp_path)
        model = SuitabilityCNN(seed=0)
        history = train(
            model,
            train_ds,
            config=TrainConfig(epochs=6, batch_size=4, learning_rate=2e-3, seed=0),
            val_dataset=val_ds,
        )
        assert history.train_loss[-1] < history.train_loss[0]
        assert history.val_loss[-1] < history.val_loss[0]
        assert len(history.train_loss) == 6
        assert len(history.val_loss) == 6
        assert history.epochs == 6

    def test_history_shapes_without_validation(self, tmp_path) -> None:
        train_ds, _ = _datasets(tmp_path)
        history = train(SuitabilityCNN(seed=0), train_ds, config=TrainConfig(epochs=2, seed=1))
        assert len(history.train_loss) == 2
        assert history.val_loss == ()

    def test_deterministic_in_seed(self, tmp_path) -> None:
        train_ds, _ = _datasets(tmp_path, per_split=6)
        config = TrainConfig(epochs=5, batch_size=4, learning_rate=2e-3, seed=42)
        first = train(SuitabilityCNN(seed=0), train_ds, config=config)
        second = train(SuitabilityCNN(seed=0), train_ds, config=config)
        assert first.train_loss == pytest.approx(second.train_loss, rel=0.0, abs=0.0)
        different = train(
            SuitabilityCNN(seed=0),
            train_ds,
            config=TrainConfig(epochs=5, batch_size=4, learning_rate=2e-3, seed=43),
        )
        assert different.train_loss != first.train_loss

    def test_empty_dataset_raises(self) -> None:
        empty = SuitabilityDataset([])
        with pytest.raises(ValueError, match="empty dataset"):
            train(SuitabilityCNN(seed=0), empty)

    def test_zero_epochs_raises(self, tmp_path) -> None:
        train_ds, _ = _datasets(tmp_path)
        with pytest.raises(ValueError, match="epochs"):
            train(SuitabilityCNN(seed=0), train_ds, config=TrainConfig(epochs=0))


class TestEvaluate:
    def test_matches_manual_computation(self, tmp_path) -> None:
        train_ds, _ = _datasets(tmp_path)
        model = SuitabilityCNN(seed=1)
        x, y = train_ds.make_batch(list(range(len(train_ds))))
        pred = model.forward(x)
        diff = pred - y
        expected = SplitMetrics(
            mse=float((diff * diff).mean()),
            mae=float(np.abs(diff).mean()),
            count=len(train_ds),
        )
        assert evaluate(model, train_ds) == expected

    def test_deterministic(self, tmp_path) -> None:
        train_ds, _ = _datasets(tmp_path)
        model = SuitabilityCNN(seed=2)
        assert evaluate(model, train_ds) == evaluate(model, train_ds)

    def test_empty_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="empty dataset"):
            evaluate(SuitabilityCNN(seed=0), SuitabilityDataset([]))

    def test_metrics_in_unit_range(self, tmp_path) -> None:
        train_ds, _ = _datasets(tmp_path)
        metrics = evaluate(SuitabilityCNN(seed=0), train_ds)
        assert 0.0 <= metrics.mse <= 1.0
        assert 0.0 <= metrics.mae <= 1.0
        assert metrics.count == len(train_ds)


def test_dataclasses_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        TrainConfig().epochs = 5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        TrainingHistory((), (), 1).epochs = 2  # type: ignore[misc]
