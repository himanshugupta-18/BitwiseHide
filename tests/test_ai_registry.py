"""
Phase 2.8.5 model registry tests.

Covers ai.registry:
- registry registration
- registry retrieval
- registry listing
- duplicate version rejection
- registry load integrity verification
- missing artifact handling
- delete operations

All tests are fully offline; no network, no BSDS500 download.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ai.cnn import SuitabilityCNN
from ai.dataloader import SuitabilityDataset
from ai.prepare_dataset import Dataset, write_synthetic_dataset
from ai.registry import ModelRegistry, RegistryEntry, RegistryError
from ai.split import Split, resolve_split
from ai.train_run import RunConfig, _run_training_with_checkpointing_internal, create_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from ai.artifact import ModelArtifact


def _make_synthetic_dataset(
    tmp_path: Path, per_split: int = 2, size: tuple[int, int] = (16, 16), seed: int = 0
) -> Dataset:
    """Create a synthetic dataset for testing."""
    return write_synthetic_dataset(tmp_path / "ds", per_split=per_split, size=size, seed=seed)


def _build_datasets(
    dataset: Dataset,
) -> tuple[SuitabilityDataset, SuitabilityDataset, SuitabilityDataset]:
    splits = resolve_split(dataset)
    return (
        SuitabilityDataset(splits[Split.TRAIN]),
        SuitabilityDataset(splits[Split.VAL]),
        SuitabilityDataset(splits[Split.TEST]),
    )


def _make_artifact(tmp_path: Path, seed: int = 0) -> tuple[ModelArtifact, Dataset]:
    """Create a test artifact."""
    dataset = _make_synthetic_dataset(tmp_path, per_split=3, seed=seed)
    cfg = RunConfig(seed=seed, epochs=2, learning_rate=1e-3, batch_size=2)
    train_ds, val_ds, test_ds = _build_datasets(dataset)
    run = _run_training_with_checkpointing_internal(train_ds, val_ds, test_ds, cfg)
    artifact = create_artifact(run.result, run.model, run_config=cfg, dataset=dataset)
    return artifact, dataset


class TestRegistryRegistration:
    """Test artifact registration in the registry."""

    def test_register_artifact(self, tmp_path: Path) -> None:
        """Register an artifact and verify entry."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path, seed=42)

        entry = registry.register(artifact, model_name="suitability_cnn", version="v1")

        assert entry.model_name == "suitability_cnn"
        assert entry.version == "v1"
        assert entry.artifact_id == artifact.metadata.artifact_id
        assert entry.path.is_dir()
        assert (entry.path / "metadata.json").is_file()
        assert (entry.path / "weights.npz").is_file()

    def test_register_overwrite_true(self, tmp_path: Path) -> None:
        """Overwrite an existing version when overwrite=True."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact1, _ = _make_artifact(tmp_path, seed=1)
        artifact2, _ = _make_artifact(tmp_path, seed=2)

        registry.register(artifact1, model_name="m", version="v1")
        entry2 = registry.register(artifact2, model_name="m", version="v1", overwrite=True)

        assert entry2.artifact_id == artifact2.metadata.artifact_id

    def test_register_duplicate_version_rejected(self, tmp_path: Path) -> None:
        """Registering duplicate version without overwrite raises."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact1, _ = _make_artifact(tmp_path, seed=1)
        artifact2, _ = _make_artifact(tmp_path, seed=2)

        registry.register(artifact1, model_name="m", version="v1")

        with pytest.raises(RegistryError, match="already exists"):
            registry.register(artifact2, model_name="m", version="v1", overwrite=False)

    def test_register_invalid_model_name(self, tmp_path: Path) -> None:
        """Invalid model name (with path separators) raises."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        with pytest.raises(RegistryError, match="invalid characters"):
            registry.register(artifact, model_name="a/b", version="v1")

    def test_register_invalid_version(self, tmp_path: Path) -> None:
        """Invalid version (with path separators) raises."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        with pytest.raises(RegistryError, match="invalid characters"):
            registry.register(artifact, model_name="m", version="v/1")

    def test_register_empty_name_rejected(self, tmp_path: Path) -> None:
        """Empty model name or version raises."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        with pytest.raises(RegistryError, match="cannot be empty"):
            registry.register(artifact, model_name="", version="v1")

        with pytest.raises(RegistryError, match="cannot be empty"):
            registry.register(artifact, model_name="m", version="")


class TestRegistryRetrieval:
    """Test artifact retrieval from the registry."""

    def test_load_artifact(self, tmp_path: Path) -> None:
        """Load a registered artifact."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path, seed=42)

        registry.register(artifact, model_name="m", version="v1")
        loaded = registry.load("m", "v1")

        assert loaded.metadata.artifact_id == artifact.metadata.artifact_id
        assert loaded.metadata.best_epoch == artifact.metadata.best_epoch

    def test_load_model(self, tmp_path: Path) -> None:
        """Load and reconstruct model from registry."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path, seed=42)

        registry.register(artifact, model_name="m", version="v1")
        model = registry.load_model("m", "v1")

        assert isinstance(model, SuitabilityCNN)
        # Model should produce valid predictions
        import numpy as np

        x = np.random.default_rng(0).random((1, 8, 8, 3))
        pred = model.forward(x)
        assert pred.shape == (1, 8, 8)
        assert pred.min() >= 0.0 and pred.max() <= 1.0

    def test_load_missing_model_raises(self, tmp_path: Path) -> None:
        """Loading non-existent model raises."""
        registry = ModelRegistry(tmp_path / "registry")

        with pytest.raises(RegistryError, match="not found"):
            registry.load("nonexistent", "v1")

    def test_load_missing_version_raises(self, tmp_path: Path) -> None:
        """Loading non-existent version raises."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)
        registry.register(artifact, model_name="m", version="v1")

        with pytest.raises(RegistryError, match="not found"):
            registry.load("m", "v2")

    def test_load_invalid_artifact_raises(self, tmp_path: Path) -> None:
        """Loading a corrupted artifact raises ArtifactError."""
        from ai.artifact import ArtifactError

        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)
        registry.register(artifact, model_name="m", version="v1")

        # Corrupt the weights
        weights_path = registry._version_dir("m", "v1") / "weights.npz"
        weights_path.write_bytes(b"corrupted")

        with pytest.raises(ArtifactError, match="Failed to load weights"):
            registry.load("m", "v1")


class TestRegistryListing:
    """Test registry listing operations."""

    def test_list_models(self, tmp_path: Path) -> None:
        """List all registered model names."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        registry.register(artifact, model_name="model_a", version="v1")
        registry.register(artifact, model_name="model_b", version="v1")

        models = registry.list_models()
        assert "model_a" in models
        assert "model_b" in models
        assert len(models) == 2

    def test_list_versions(self, tmp_path: Path) -> None:
        """List versions for a specific model."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        registry.register(artifact, model_name="m", version="v1")
        registry.register(artifact, model_name="m", version="v2")
        registry.register(artifact, model_name="m", version="v3")

        versions = registry.list_versions("m")
        assert versions == ["v1", "v2", "v3"]

    def test_list_versions_empty_for_unknown_model(self, tmp_path: Path) -> None:
        """List versions for unknown model returns empty list."""
        registry = ModelRegistry(tmp_path / "registry")
        versions = registry.list_versions("unknown")
        assert versions == []

    def test_list_entries(self, tmp_path: Path) -> None:
        """List all entries with metadata."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        registry.register(artifact, model_name="m", version="v1", description="First version")
        registry.register(artifact, model_name="m", version="v2", description="Second version")

        entries = registry.list_entries()
        assert len(entries) == 2
        assert all(isinstance(e, RegistryEntry) for e in entries)
        assert entries[0].model_name == "m"
        assert entries[0].version in ("v1", "v2")

    def test_exists(self, tmp_path: Path) -> None:
        """Check if model version exists."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        assert not registry.exists("m", "v1")
        registry.register(artifact, model_name="m", version="v1")
        assert registry.exists("m", "v1")
        assert not registry.exists("m", "v2")


class TestRegistryDeletion:
    """Test registry deletion operations."""

    def test_delete_version(self, tmp_path: Path) -> None:
        """Delete a specific version."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        registry.register(artifact, model_name="m", version="v1")
        assert registry.exists("m", "v1")

        registry.delete("m", "v1")
        assert not registry.exists("m", "v1")

    def test_delete_version_removes_empty_model_dir(self, tmp_path: Path) -> None:
        """Deleting last version removes model directory."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        registry.register(artifact, model_name="m", version="v1")
        model_dir = registry._model_dir("m")
        assert model_dir.is_dir()

        registry.delete("m", "v1")
        assert not model_dir.exists()

    def test_delete_version_keeps_model_dir_if_other_versions_exist(self, tmp_path: Path) -> None:
        """Deleting one version keeps model directory if others exist."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        registry.register(artifact, model_name="m", version="v1")
        registry.register(artifact, model_name="m", version="v2")

        registry.delete("m", "v1")
        assert registry._model_dir("m").is_dir()
        assert registry.exists("m", "v2")

    def test_delete_missing_version_raises(self, tmp_path: Path) -> None:
        """Deleting non-existent version raises."""
        registry = ModelRegistry(tmp_path / "registry")

        with pytest.raises(RegistryError, match="not found"):
            registry.delete("m", "v1")

    def test_delete_model(self, tmp_path: Path) -> None:
        """Delete all versions of a model."""
        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        registry.register(artifact, model_name="m", version="v1")
        registry.register(artifact, model_name="m", version="v2")

        registry.delete_model("m")
        assert not registry.exists("m", "v1")
        assert not registry.exists("m", "v2")
        assert not registry._model_dir("m").exists()

    def test_delete_missing_model_raises(self, tmp_path: Path) -> None:
        """Deleting non-existent model raises."""
        registry = ModelRegistry(tmp_path / "registry")

        with pytest.raises(RegistryError, match="not found"):
            registry.delete_model("unknown")


class TestRegistryIntegrity:
    """Test registry integrity verification on load."""

    def test_load_verifies_checksum(self, tmp_path: Path) -> None:
        """Registry load verifies artifact checksum."""
        import numpy as np

        from ai.artifact import ArtifactError

        registry = ModelRegistry(tmp_path / "registry")
        artifact, _ = _make_artifact(tmp_path)

        registry.register(artifact, model_name="m", version="v1")

        # Corrupt the weights file directly
        weights_path = registry._version_dir("m", "v1") / "weights.npz"
        npz = np.load(weights_path)
        data = {k: v.copy() for k, v in npz.items()}
        data["weight_0"] = data["weight_0"] * 1.001
        np.savez_compressed(weights_path, **data)

        with pytest.raises(ArtifactError, match="Checksum mismatch"):
            registry.load("m", "v1")


class TestRegistryPersistence:
    """Test registry persists across instances."""

    def test_registry_persists_across_instances(self, tmp_path: Path) -> None:
        """Registry data persists when creating new registry instance."""
        artifact, _ = _make_artifact(tmp_path, seed=42)

        registry1 = ModelRegistry(tmp_path / "registry")
        registry1.register(artifact, model_name="m", version="v1")

        registry2 = ModelRegistry(tmp_path / "registry")
        entry = registry2.load("m", "v1")

        assert entry.metadata.artifact_id == artifact.metadata.artifact_id
