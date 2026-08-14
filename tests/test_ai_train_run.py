"""
Phase 2.8.4 training pipeline tests.

Covers ai.train_run, fully offline:
- deterministic training with identical seeds/configuration
- training changes model parameters
- training loss/metrics are finite
- validation metrics are produced
- test metrics are produced without being used for model selection
- best-validation-model selection
- synthetic end-to-end smoke training
- train/validation/test leakage remains impossible
- invalid training configuration fails clearly

All datasets are synthetic and generated in memory / in tmp_path.
No network, no BSDS500 download, no ML framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ai.cnn import SuitabilityCNN
from ai.dataloader import SuitabilityDataset
from ai.prepare_dataset import DatasetError, Split, write_synthetic_dataset
from ai.split import resolve_split
from ai.train_run import (
    RunConfig,
    RunResult,
    run_real_training,
    run_synthetic_training,
)
from ai.training import (
    TrainConfig,
    restore_parameters,
    snapshot_parameters,
    train,
)


def _flat(model: SuitabilityCNN) -> np.ndarray:
    """All parameters as one flat array: per layer, weight then bias."""
    return np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in model.parameters()])


class TestDeterminism:
    """Same seed and config must reproduce the same run result."""

    def test_identical_seeds_reproduce(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=1, epochs=4, learning_rate=1e-3, batch_size=4)
        # Fixed tmp root so both runs use identical dataset bytes.
        run1 = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run1"
        )
        run2 = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run2"
        )
        assert run1.history.train_loss == run2.history.train_loss
        assert run1.history.val_loss == run2.history.val_loss
        assert run1.best_epoch == run2.best_epoch
        assert run1.best_val_mse == pytest.approx(run2.best_val_mse, rel=0.0, abs=0.0)
        assert run1.test_mse == pytest.approx(run2.test_mse, rel=0.0, abs=0.0)
        assert run1.test_mae == pytest.approx(run2.test_mae, rel=0.0, abs=0.0)

    def test_different_seed_differs(self, tmp_path: Path) -> None:
        cfg_a = RunConfig(seed=1, epochs=4, learning_rate=1e-3, batch_size=4)
        cfg_b = RunConfig(seed=2, epochs=4, learning_rate=1e-3, batch_size=4)
        run_a = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg_a, tmp_root=tmp_path / "a"
        )
        run_b = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg_b, tmp_root=tmp_path / "b"
        )
        # Training starts from different initialization, so histories differ.
        assert run_a.history.train_loss != run_b.history.train_loss


class TestParameterUpdates:
    """Training must change the model parameters."""

    def test_training_changes_parameters(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=0, epochs=3, learning_rate=1e-3, batch_size=4)
        result = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
        )
        # Compare the trained best model against a fresh untrained model.
        fresh = SuitabilityCNN(seed=cfg.seed)
        trained = SuitabilityCNN(seed=cfg.seed)
        # Restore the selected (best) model state into a fresh instance.
        # We reconstruct by training again and snapping — simpler: compare via
        # the fact that loss decreased means params moved. Validate indirectly:
        # the best validation MSE must be less than the initial random MSE.
        # Direct check: re-evaluate a fresh model on val.
        dataset = write_synthetic_dataset(tmp_path / "ds", per_split=3, size=(16, 16), seed=0)
        splits = resolve_split(dataset)
        val_ds = SuitabilityDataset(splits[Split.VAL])
        init_metrics = evaluate(fresh, val_ds)
        assert result.best_val_mse < init_metrics.mse or result.best_val_mse <= init_metrics.mse
        # The trained model differs from the fresh initialization.
        # We need to actually train the `trained` model to compare.
        train_ds = SuitabilityDataset(splits[Split.TRAIN])
        train(
            trained,
            train_ds,
            config=TrainConfig(
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                seed=cfg.seed,
            ),
            val_dataset=val_ds,
        )
        assert not np.allclose(_flat(trained), _flat(fresh))


class TestMetrics:
    """All reported metrics must be finite and in range."""

    def test_losses_are_finite(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=0, epochs=5, learning_rate=1e-3, batch_size=4)
        result = run_synthetic_training(
            per_split=4, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
        )
        for loss in result.history.train_loss:
            assert np.isfinite(loss)
        for loss in result.history.val_loss:
            assert np.isfinite(loss)
        assert np.isfinite(result.best_val_mse)
        assert np.isfinite(result.best_val_mae)
        assert np.isfinite(result.test_mse)
        assert np.isfinite(result.test_mae)

    def test_metrics_in_unit_range(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=0, epochs=5, learning_rate=1e-3, batch_size=4)
        result = run_synthetic_training(
            per_split=4, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
        )
        # MSE/MAE for a [0,1] target and [0,1] prediction are bounded by 1.
        assert 0.0 <= result.best_val_mse <= 1.0
        assert 0.0 <= result.best_val_mae <= 1.0
        assert 0.0 <= result.test_mse <= 1.0
        assert 0.0 <= result.test_mae <= 1.0

    def test_validation_metrics_produced(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=0, epochs=5, learning_rate=1e-3, batch_size=4)
        result = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
        )
        # Validation loss recorded per epoch.
        assert len(result.history.val_loss) == cfg.epochs
        # Best validation metrics populated.
        assert result.best_epoch >= 0
        assert result.best_epoch < cfg.epochs
        assert result.best_val_mse >= 0.0

    def test_test_metrics_produced(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=0, epochs=5, learning_rate=1e-3, batch_size=4)
        result = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
        )
        # Test metrics produced and report the sample count.
        assert result.test_count > 0
        assert np.isfinite(result.test_mse)
        assert np.isfinite(result.test_mae)


class TestBestModelSelection:
    """Best model must be selected by validation performance only."""

    def test_best_model_from_validation(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=1, epochs=6, learning_rate=1e-3, batch_size=4)
        result = run_synthetic_training(
            per_split=4, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
        )
        # The best epoch's validation loss equals the recorded best validation MSE.
        val_at_best = result.history.val_loss[result.best_epoch]
        assert result.best_val_mse == pytest.approx(val_at_best, rel=0.0, abs=1e-12)
        # The best validation MSE is the minimum across all validation losses.
        val_min = min(result.history.val_loss)
        assert result.best_val_mse == pytest.approx(val_min, rel=0.0, abs=1e-12)

    def test_test_not_used_for_selection(self, tmp_path: Path) -> None:
        """Test performance must not be the selection criterion.

        We verify the selected model is the validation-best by confirming
        restoring the best snapshot yields exactly the reported test metrics and
        that the selection epoch is driven by val_loss, not test_loss.
        """
        cfg = RunConfig(seed=3, epochs=5, learning_rate=1e-3, batch_size=4)
        result = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
        )
        # Reconstruct the best model and confirm test metrics match exactly.
        dataset = write_synthetic_dataset(tmp_path / "ds", per_split=3, size=(16, 16), seed=3)
        splits = resolve_split(dataset)
        train_ds = SuitabilityDataset(splits[Split.TRAIN])
        val_ds = SuitabilityDataset(splits[Split.VAL])
        test_ds = SuitabilityDataset(splits[Split.TEST])

        model = SuitabilityCNN(seed=cfg.seed)
        # Re-train and capture snapshots to reconstruct exact best model.
        from ai.training import train

        best: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
        best_mse = float("inf")

        def cb(_epoch: int, m: SuitabilityCNN) -> None:
            nonlocal best, best_mse
            from ai.training import evaluate as _eval

            mse = _eval(m, val_ds).mse
            if mse < best_mse:
                best_mse = mse
                best = snapshot_parameters(m)

        train(model, train_ds, config=cfg, val_dataset=val_ds, epoch_callback=cb)  # type: ignore[arg-type]
        assert best is not None
        restore_parameters(model, best)
        from ai.training import evaluate as _eval2

        recomputed = _eval2(model, test_ds)
        assert recomputed.mse == pytest.approx(result.test_mse, rel=0.0, abs=1e-12)
        assert recomputed.mae == pytest.approx(result.test_mae, rel=0.0, abs=1e-12)


class TestSyntheticSmoke:
    """The offline synthetic path must exercise the full pipeline quickly."""

    def test_end_to_end_smoke(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=0, epochs=3, learning_rate=1e-3, batch_size=2)
        result = run_synthetic_training(
            per_split=2, size=(32, 32), run_config=cfg, tmp_root=tmp_path / "smoke"
        )
        # All three splits were used.
        assert len(result.history.train_loss) == cfg.epochs
        assert len(result.history.val_loss) == cfg.epochs
        assert result.test_count == 2
        # Loss should be non-negative and finite.
        assert result.history.train_loss[-1] >= 0.0
        assert result.test_mse >= 0.0

    def test_no_leakage_across_splits(self, tmp_path: Path) -> None:
        """Train/val/test sets must remain disjoint at the image level."""
        cfg = RunConfig(seed=0, epochs=2, learning_rate=1e-3, batch_size=2)
        run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "leak"
        )
        # The synthetic dataset's splits are image-disjoint by construction.
        dataset = write_synthetic_dataset(tmp_path / "ds", per_split=3, size=(16, 16), seed=0)
        splits = resolve_split(dataset)
        train_ids = {info.image_id for info in splits[Split.TRAIN]}
        val_ids = {info.image_id for info in splits[Split.VAL]}
        test_ids = {info.image_id for info in splits[Split.TEST]}
        assert train_ids & val_ids == set()
        assert train_ids & test_ids == set()
        assert val_ids & test_ids == set()
        assert len(train_ids | val_ids | test_ids) == 9


class TestRealDataset:
    """Real training requires a local root and fails clearly on missing data."""

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        cfg = RunConfig(seed=0, epochs=2, training_size=(64, 64))
        with pytest.raises(DatasetError, match="does not exist"):
            run_real_training(tmp_path / "nope", run_config=cfg)

    def test_real_training_runs_on_local_synthetic_layout(self, tmp_path: Path) -> None:
        """Use a locally written synthetic dataset (BSDS500-like layout) to
        exercise the real-training path without downloading BSDS500."""
        # Write a synthetic dataset with the official BSDS500 folder layout.
        root = tmp_path / "local_bsds"
        write_synthetic_dataset(root, per_split=2, size=(64, 64), seed=0)
        cfg = RunConfig(seed=0, epochs=2, learning_rate=1e-3, batch_size=2, training_size=(64, 64))
        result = run_real_training(root, run_config=cfg)
        assert len(result.history.train_loss) == cfg.epochs
        assert result.test_count == 2

    def test_real_training_requires_training_size(self, tmp_path: Path) -> None:
        """Mixed-size real datasets must set training_size or fail clearly."""
        root = tmp_path / "local_bsds"
        write_synthetic_dataset(root, per_split=2, size=(64, 64), seed=0)
        cfg = RunConfig(seed=0, epochs=2)  # No training_size.
        with pytest.raises(ValueError, match="training_size"):
            run_real_training(root, run_config=cfg)


class TestInvalidConfig:
    """Invalid configurations must fail clearly, not silently misbehave."""

    def test_zero_epochs_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="epochs"):
            cfg = RunConfig(seed=0, epochs=0, learning_rate=1e-3, batch_size=4)
            run_synthetic_training(
                per_split=2, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
            )

    def test_negative_learning_rate_invalid(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            cfg = RunConfig(seed=0, epochs=2, learning_rate=-1e-3, batch_size=4)
            run_synthetic_training(
                per_split=2, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "run"
            )


def test_run_result_is_reproducible(tmp_path: Path) -> None:
    """Two runs with the same config and dataset produce identical RunResults."""
    cfg = RunConfig(seed=5, epochs=3, learning_rate=1e-3, batch_size=4)
    r1 = run_synthetic_training(
        per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "r1"
    )
    r2 = run_synthetic_training(
        per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path / "r2"
    )
    assert isinstance(r1, RunResult)
    assert isinstance(r2, RunResult)
    assert r1.best_epoch == r2.best_epoch
    assert r1.best_val_mse == pytest.approx(r2.best_val_mse, rel=0.0, abs=0.0)
    assert r1.test_mse == pytest.approx(r2.test_mse, rel=0.0, abs=0.0)
    assert r1.test_mae == pytest.approx(r2.test_mae, rel=0.0, abs=0.0)


# Imported at module level to avoid circular imports in TYPE_CHECKING blocks.
from ai.training import evaluate  # noqa: E402
