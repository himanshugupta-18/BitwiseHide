"""
Phase 2.8.5 model artifact tests.

Covers ai.artifact:
- artifact creation from trained model
- artifact round-trip save/load
- model parameters preserved after round-trip
- metadata preserved
- deterministic artifact representation where practical
- checksum/integrity verification
- corrupted artifact rejection
- invalid schema rejection
- invalid parameter shape rejection
- non-finite parameter rejection

All tests are fully offline; no network, no BSDS500 download.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ai.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    ModelArtifact,
)
from ai.cnn import SuitabilityCNN
from ai.dataloader import SuitabilityDataset
from ai.prepare_dataset import Dataset, write_synthetic_dataset
from ai.split import Split, resolve_split
from ai.train_run import RunConfig, create_artifact, run_synthetic_training
from ai.training import SplitMetrics, TrainConfig, evaluate, train

if TYPE_CHECKING:
    from pathlib import Path


def _make_synthetic_dataset(
    tmp_path: Path, per_split: int = 2, size: tuple[int, int] = (16, 16), seed: int = 0
) -> Dataset:
    """Create a synthetic dataset for testing."""
    return write_synthetic_dataset(tmp_path / "ds", per_split=per_split, size=size, seed=seed)


def _make_train_result(dataset: Dataset, _tmp_path: Path, seed: int = 0) -> tuple:
    """Run a quick training and return (model, result, dataset)."""
    cfg = RunConfig(seed=seed, epochs=2, learning_rate=1e-3, batch_size=2)
    train_ds, val_ds, test_ds = _build_datasets(dataset)
    model = SuitabilityCNN(seed=seed)

    best_val_mse = float("inf")
    best_snapshot: list[tuple[np.ndarray, np.ndarray]] | None = None
    best_epoch = -1

    def epoch_callback(epoch: int, current_model: SuitabilityCNN) -> None:
        nonlocal best_val_mse, best_snapshot, best_epoch
        val_metrics = evaluate(current_model, val_ds)
        if val_metrics.mse < best_val_mse:
            best_val_mse = val_metrics.mse
            # Use snapshot_parameters to get proper copy
            from ai.training import snapshot_parameters
            best_snapshot = list(snapshot_parameters(current_model))
            best_epoch = epoch

    history = train(
        model,
        train_ds,
        config=TrainConfig(
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
        ),
        val_dataset=val_ds,
        epoch_callback=epoch_callback,
    )

    from ai.training import restore_parameters

    # best_snapshot is guaranteed to be set since we run at least 1 epoch
    assert best_snapshot is not None
    restore_parameters(model, best_snapshot)
    best_val_metrics = evaluate(model, val_ds)
    test_metrics = evaluate(model, test_ds)

    from ai.train_run import RunResult

    result = RunResult(
        history=history,
        best_epoch=best_epoch,
        best_val_mse=best_val_metrics.mse,
        best_val_mae=best_val_metrics.mae,
        test_mse=test_metrics.mse,
        test_mae=test_metrics.mae,
        test_count=test_metrics.count,
    )
    return model, result


def _build_datasets(
    dataset: Dataset,
) -> tuple[SuitabilityDataset, SuitabilityDataset, SuitabilityDataset]:
    splits = resolve_split(dataset)
    return (
        SuitabilityDataset(splits[Split.TRAIN]),
        SuitabilityDataset(splits[Split.VAL]),
        SuitabilityDataset(splits[Split.TEST]),
    )


class TestArtifactCreation:
    """Test artifact creation from training results."""

    def test_create_artifact_from_training(self, tmp_path: Path) -> None:
        """Create artifact from a synthetic training run."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=3, seed=42)
        model, result = _make_train_result(dataset, tmp_path, seed=42)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=42),
            seed=42,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        assert artifact.metadata.artifact_id != ""
        assert artifact.metadata.schema_version == ARTIFACT_SCHEMA_VERSION
        assert artifact.metadata.architecture.name == "suitability_cnn_v1"
        assert artifact.metadata.architecture.depth == 5
        assert artifact.metadata.seed == 42
        assert artifact.metadata.best_epoch == result.best_epoch
        assert artifact.metadata.best_val_metrics.mse == pytest.approx(result.best_val_mse)
        assert artifact.metadata.test_metrics.mse == pytest.approx(result.test_mse)

    def test_create_artifact_deterministic_id(self, tmp_path: Path) -> None:
        """Same training config + seed + dataset produces same artifact_id."""
        dataset1 = _make_synthetic_dataset(tmp_path / "d1", per_split=2, seed=0)
        dataset2 = _make_synthetic_dataset(tmp_path / "d2", per_split=2, seed=0)

        model1, result1 = _make_train_result(dataset1, tmp_path / "r1", seed=0)
        model2, result2 = _make_train_result(dataset2, tmp_path / "r2", seed=0)

        artifact1 = ModelArtifact.create(
            model=model1,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset1,
            history=result1.history,
            best_epoch=result1.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result1.best_val_mse, mae=result1.best_val_mae, count=len(dataset1.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result1.test_mse, mae=result1.test_mae, count=result1.test_count
            ),
        )

        artifact2 = ModelArtifact.create(
            model=model2,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset2,
            history=result2.history,
            best_epoch=result2.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result2.best_val_mse, mae=result2.best_val_mae, count=len(dataset2.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result2.test_mse, mae=result2.test_mae, count=result2.test_count
            ),
        )

        # Artifact IDs should match (same seed, config, dataset structure)
        assert artifact1.metadata.artifact_id == artifact2.metadata.artifact_id

    def test_artifact_parameters_count_matches_model(self, tmp_path: Path) -> None:
        """Artifact parameter count matches model's actual parameter count."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=1)
        model, result = _make_train_result(dataset, tmp_path, seed=1)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=1),
            seed=1,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        model_param_count = sum(w.size + b.size for w, b in model.parameters())
        assert artifact.metadata.architecture.parameter_count == model_param_count


class TestArtifactRoundTrip:
    """Test artifact save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Save artifact to disk and load it back."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=3, seed=42)
        model, result = _make_train_result(dataset, tmp_path, seed=42)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=42),
            seed=42,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        loaded = ModelArtifact.load(artifact_dir)

        # Metadata preserved
        assert loaded.metadata.artifact_id == artifact.metadata.artifact_id
        assert loaded.metadata.schema_version == artifact.metadata.schema_version
        assert loaded.metadata.seed == artifact.metadata.seed
        assert loaded.metadata.best_epoch == artifact.metadata.best_epoch
        assert loaded.metadata.best_val_metrics.mse == pytest.approx(
            artifact.metadata.best_val_metrics.mse
        )
        assert loaded.metadata.test_metrics.mse == pytest.approx(artifact.metadata.test_metrics.mse)
        assert loaded.metadata.dataset.name == artifact.metadata.dataset.name

        # Parameters preserved exactly
        for (w1, b1), (w2, b2) in zip(artifact.parameters, loaded.parameters, strict=True):
            assert np.allclose(w1, w2)
            assert np.allclose(b1, b2)

    def test_reconstructed_model_matches_original(self, tmp_path: Path) -> None:
        """Reconstructed model produces identical predictions."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=3, seed=123)
        model, result = _make_train_result(dataset, tmp_path, seed=123)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=123),
            seed=123,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        loaded = ModelArtifact.load(artifact_dir)
        reconstructed = loaded.reconstruct_model()

        # Test on validation data
        _, val_ds, test_ds = _build_datasets(dataset)
        orig_val = evaluate(model, val_ds)
        recon_val = evaluate(reconstructed, val_ds)
        assert orig_val.mse == pytest.approx(recon_val.mse, abs=1e-12)
        assert orig_val.mae == pytest.approx(recon_val.mae, abs=1e-12)

        orig_test = evaluate(model, test_ds)
        recon_test = evaluate(reconstructed, test_ds)
        assert orig_test.mse == pytest.approx(recon_test.mse, abs=1e-12)
        assert orig_test.mae == pytest.approx(recon_test.mae, abs=1e-12)


class TestArtifactValidation:
    """Test artifact validation on load."""

    def test_invalid_schema_version_rejected(self, tmp_path: Path) -> None:
        """Loading artifact with wrong schema version fails."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        model, result = _make_train_result(dataset, tmp_path, seed=0)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        # Corrupt metadata: change schema version
        metadata_path = artifact_dir / "metadata.json"
        data = json.loads(metadata_path.read_text())
        data["schema_version"] = 999
        metadata_path.write_text(json.dumps(data))

        with pytest.raises(ArtifactError, match="Unsupported schema version"):
            ModelArtifact.load(artifact_dir)

    def test_missing_metadata_rejected(self, tmp_path: Path) -> None:
        """Missing metadata.json fails clearly."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        weights_path = artifact_dir / "weights.npz"
        np.savez(weights_path, weight_0=np.zeros((1,)), bias_0=np.zeros((1,)))

        with pytest.raises(ArtifactError, match="Missing metadata.json"):
            ModelArtifact.load(artifact_dir)

    def test_missing_weights_rejected(self, tmp_path: Path) -> None:
        """Missing weights.npz fails clearly."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        metadata_path = artifact_dir / "metadata.json"
        metadata_path.write_text(json.dumps({"schema_version": 1}))

        with pytest.raises(ArtifactError, match="Missing weights.npz"):
            ModelArtifact.load(artifact_dir)

    def test_corrupted_weights_rejected(self, tmp_path: Path) -> None:
        """Corrupted weights.npz fails."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        model, result = _make_train_result(dataset, tmp_path, seed=0)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        # Corrupt weights file
        weights_path = artifact_dir / "weights.npz"
        weights_path.write_bytes(b"not a valid npz file")

        with pytest.raises(ArtifactError, match="Failed to load weights.npz"):
            ModelArtifact.load(artifact_dir)

    def test_invalid_parameter_shapes_rejected(self, tmp_path: Path) -> None:
        """Wrong parameter shapes in weights.npz fails."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        model, result = _make_train_result(dataset, tmp_path, seed=0)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        # Corrupt weights: change shape of weight_0
        weights_path = artifact_dir / "weights.npz"
        npz = np.load(weights_path)
        data = {k: v.copy() for k, v in npz.items()}
        data["weight_0"] = np.zeros((3, 3, 3, 10))  # Wrong output channels
        np.savez_compressed(weights_path, **data)

        with pytest.raises(ArtifactError, match="weight shape"):
            ModelArtifact.load(artifact_dir)

    def test_non_finite_parameters_rejected(self, tmp_path: Path) -> None:
        """Non-finite parameter values (NaN/inf) fail."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        model, result = _make_train_result(dataset, tmp_path, seed=0)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        # Corrupt weights: inject NaN
        weights_path = artifact_dir / "weights.npz"
        npz = np.load(weights_path)
        data = {k: v.copy() for k, v in npz.items()}
        data["weight_0"][0, 0, 0, 0] = np.nan
        np.savez_compressed(weights_path, **data)

        with pytest.raises(ArtifactError, match="non-finite values"):
            ModelArtifact.load(artifact_dir)

    def test_checksum_mismatch_rejected(self, tmp_path: Path) -> None:
        """Modified weights.npz fails checksum verification."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        model, result = _make_train_result(dataset, tmp_path, seed=0)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        # Modify weights file without updating metadata checksum
        weights_path = artifact_dir / "weights.npz"
        npz = np.load(weights_path)
        data = {k: v.copy() for k, v in npz.items()}
        data["weight_0"] = data["weight_0"] * 1.001  # Slight modification
        np.savez_compressed(weights_path, **data)

        with pytest.raises(ArtifactError, match="Checksum mismatch"):
            ModelArtifact.load(artifact_dir)


class TestMetadataJSON:
    """Test metadata JSON serialization."""

    def test_metadata_json_contains_all_fields(self, tmp_path: Path) -> None:
        """Saved metadata.json has all expected fields."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        model, result = _make_train_result(dataset, tmp_path, seed=0)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
            description="Test artifact",
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        metadata_path = artifact_dir / "metadata.json"
        data = json.loads(metadata_path.read_text())

        required_fields = [
            "schema_version",
            "artifact_id",
            "architecture",
            "train_config",
            "seed",
            "dataset",
            "history",
            "best_epoch",
            "best_val_metrics",
            "test_metrics",
            "weights_checksum",
            "created_at",
            "description",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        assert data["description"] == "Test artifact"

    def test_weights_npz_contains_all_layers(self, tmp_path: Path) -> None:
        """Saved weights.npz has all layer weights and biases."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        model, result = _make_train_result(dataset, tmp_path, seed=0)

        artifact = ModelArtifact.create(
            model=model,
            train_config=TrainConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            seed=0,
            dataset=dataset,
            history=result.history,
            best_epoch=result.best_epoch,
            best_val_metrics=SplitMetrics(
                mse=result.best_val_mse, mae=result.best_val_mae, count=len(dataset.images) // 3
            ),
            test_metrics=SplitMetrics(
                mse=result.test_mse, mae=result.test_mae, count=result.test_count
            ),
        )

        artifact_dir = tmp_path / "artifact"
        artifact.save(artifact_dir)

        weights_path = artifact_dir / "weights.npz"
        npz = np.load(weights_path)

        # 5 layers -> weight_0..4, bias_0..4
        for i in range(5):
            assert f"weight_{i}" in npz
            assert f"bias_{i}" in npz


class TestCreateArtifactFunction:
    """Test the create_artifact convenience function in train_run.py."""

    def test_create_artifact_from_run_result(self, tmp_path: Path) -> None:
        """create_artifact works with RunResult from run_synthetic_training."""
        cfg = RunConfig(seed=42, epochs=2, learning_rate=1e-3, batch_size=2)
        result = run_synthetic_training(
            per_split=3, size=(16, 16), run_config=cfg, tmp_root=tmp_path
        )

        # Need to re-run to get the model and dataset for artifact creation
        # (In practice you'd use _run_training_with_checkpointing_internal)
        dataset = _make_synthetic_dataset(tmp_path / "ds2", per_split=3, seed=42)
        from ai.train_run import _run_training_with_checkpointing_internal

        train_ds, val_ds, test_ds = _build_datasets(dataset)
        run = _run_training_with_checkpointing_internal(train_ds, val_ds, test_ds, cfg)

        artifact = create_artifact(
            result=run.result,
            model=run.model,
            run_config=cfg,
            dataset=dataset,
        )

        assert artifact.metadata.artifact_id != ""
        assert artifact.metadata.best_epoch == result.best_epoch
        assert artifact.metadata.best_val_metrics.mse == pytest.approx(result.best_val_mse)
        assert artifact.metadata.test_metrics.mse == pytest.approx(result.test_mse)

    def test_create_artifact_custom_id_and_description(self, tmp_path: Path) -> None:
        """create_artifact respects custom artifact_id and description."""
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        model, result = _make_train_result(dataset, tmp_path, seed=0)

        artifact = create_artifact(
            result=result,
            model=model,
            run_config=RunConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=0),
            dataset=dataset,
            artifact_id="custom-id-123",
            description="My custom model",
        )

        assert artifact.metadata.artifact_id == "custom-id-123"
        assert artifact.metadata.description == "My custom model"
