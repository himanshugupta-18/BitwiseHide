"""
Model artifact format and validation (Phase 2.8.5).

This module provides a deterministic, inspectable artifact format for trained
SuitabilityCNN models. The artifact consists of:

- JSON metadata (architecture, training config, seed, dataset provenance,
  selected epoch, metrics, checksums)
- NumPy .npz weights (weight and bias arrays per layer)

No pickle is used. Loading validates schema version, required metadata,
parameter shapes, finite values, and checksum integrity.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from ai.cnn import SuitabilityCNN
from ai.training import SplitMetrics, TrainConfig, TrainingHistory

if TYPE_CHECKING:
    from pathlib import Path

    from ai.prepare_dataset import Dataset

#: Current artifact schema version. Increment on breaking changes.
ARTIFACT_SCHEMA_VERSION = 1

#: Filenames inside an artifact directory.
_METADATA_FILENAME = "metadata.json"
_WEIGHTS_FILENAME = "weights.npz"


class ArtifactError(Exception):
    """Raised when an artifact is invalid, corrupted, or fails validation."""


@dataclass(frozen=True)
class ModelArchitecture:
    """Model architecture specification for reconstruction."""

    #: Architecture identifier (must match a registered architecture).
    name: str = "suitability_cnn_v1"
    #: Conv layers as (in_channels, out_channels, kernel_size).
    conv_layers: tuple[tuple[int, int, int], ...] = (
        (3, 16, 5),
        (16, 16, 3),
        (16, 32, 3),
        (32, 32, 3),
        (32, 1, 1),
    )
    #: Total number of conv layers.
    depth: int = 5
    #: Total parameter count (weights + biases).
    parameter_count: int = 0  # Computed on creation


@dataclass(frozen=True)
class DatasetProvenance:
    """Dataset identity and provenance for reproducibility."""

    #: Dataset name (e.g., "bsds500", "synthetic").
    name: str
    #: Dataset root path (as string for JSON serialization).
    root: str
    #: Source URL if applicable.
    source_url: str = ""
    #: Citation if applicable.
    citation: str = ""
    #: Total image count.
    total_images: int = 0
    #: Per-split image counts.
    split_counts: dict[str, int] | None = None
    #: Whether official BSDS500 split was available.
    official_split_available: bool = False

    def __post_init__(self) -> None:
        if self.split_counts is None:
            object.__setattr__(self, "split_counts", {})


@dataclass(frozen=True)
class ArtifactMetadata:
    """Complete metadata for a model artifact."""

    #: Artifact schema version.
    schema_version: int
    #: Unique artifact identifier (UUID4 hex or deterministic hash).
    artifact_id: str
    #: Model architecture specification.
    architecture: ModelArchitecture
    #: Training configuration used.
    train_config: TrainConfig
    #: Global random seed used for this run.
    seed: int
    #: Dataset provenance.
    dataset: DatasetProvenance
    #: Training history (per-epoch losses).
    history: TrainingHistory
    #: Selected epoch (best validation).
    best_epoch: int
    #: Validation metrics at best epoch.
    best_val_metrics: SplitMetrics
    #: Test metrics for the best-validation model.
    test_metrics: SplitMetrics
    #: SHA256 checksum of the weights.npz file.
    weights_checksum: str
    #: ISO 8601 creation timestamp (UTC).
    created_at: str
    #: Optional human-readable description.
    description: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> ArtifactMetadata:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            schema_version=data["schema_version"],
            artifact_id=data["artifact_id"],
            architecture=ModelArchitecture(**data["architecture"]),
            train_config=TrainConfig(**data["train_config"]),
            seed=data["seed"],
            dataset=DatasetProvenance(**data["dataset"]),
            history=TrainingHistory(**data["history"]),
            best_epoch=data["best_epoch"],
            best_val_metrics=SplitMetrics(**data["best_val_metrics"]),
            test_metrics=SplitMetrics(**data["test_metrics"]),
            weights_checksum=data["weights_checksum"],
            created_at=data["created_at"],
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class ModelArtifact:
    """Complete model artifact: metadata + parameters."""

    metadata: ArtifactMetadata
    #: Parameters as (weight, bias) tuples per layer, matching SuitabilityCNN.parameters()
    parameters: tuple[tuple[np.ndarray, np.ndarray], ...]

    @classmethod
    def create(
        cls,
        model: SuitabilityCNN,
        *,
        train_config: TrainConfig,
        seed: int,
        dataset: Dataset,
        history: TrainingHistory,
        best_epoch: int,
        best_val_metrics: SplitMetrics,
        test_metrics: SplitMetrics,
        artifact_id: str | None = None,
        description: str = "",
    ) -> ModelArtifact:
        """
        Create a ModelArtifact from a trained model and training results.

        Args:
            model: Trained SuitabilityCNN (already restored to best epoch).
            train_config: Training configuration used.
            seed: Global seed used for this run.
            dataset: Dataset the model was trained on.
            history: Per-epoch training history.
            best_epoch: Zero-based index of the best validation epoch.
            best_val_metrics: Validation metrics at best epoch.
            test_metrics: Test metrics for the best-validation model.
            artifact_id: Optional explicit artifact ID; generated if omitted.
            description: Optional human-readable description.

        Returns:
            A complete ModelArtifact ready to save.
        """
        from ai.training import snapshot_parameters

        parameters = snapshot_parameters(model)

        # Compute architecture parameter count
        arch = ModelArchitecture()
        param_count = sum(w.size + b.size for w, b in parameters)
        arch = ModelArchitecture(
            name=arch.name,
            conv_layers=arch.conv_layers,
            depth=arch.depth,
            parameter_count=param_count,
        )

        # Dataset provenance
        dataset_prov = DatasetProvenance(
            name=dataset.name,
            root=str(dataset.root),
            source_url=dataset.source_url,
            citation=dataset.citation,
            total_images=dataset.size,
            split_counts={split.value: count for split, count in dataset.counts_by_split().items()},
            official_split_available=dataset.official_split_available,
        )

        # Generate artifact ID if not provided
        if artifact_id is None:
            artifact_id = _generate_artifact_id(model, train_config, seed, dataset)

        # Weights checksum (computed after saving weights, but we can compute from params)
        # We'll compute it as SHA256 of the concatenated parameter bytes
        weights_bytes = b"".join(w.tobytes() + b.tobytes() for w, b in parameters)
        weights_checksum = hashlib.sha256(weights_bytes).hexdigest()

        metadata = ArtifactMetadata(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            artifact_id=artifact_id,
            architecture=arch,
            train_config=train_config,
            seed=seed,
            dataset=dataset_prov,
            history=history,
            best_epoch=best_epoch,
            best_val_metrics=best_val_metrics,
            test_metrics=test_metrics,
            weights_checksum=weights_checksum,
            created_at=_utc_now_iso(),
            description=description,
        )

        return cls(metadata=metadata, parameters=parameters)

    def save(self, directory: Path) -> None:
        """
        Save artifact to a directory.

        Creates:
        - directory/metadata.json
        - directory/weights.npz
        """
        directory.mkdir(parents=True, exist_ok=True)

        # Save weights as .npz FIRST (to compute checksum from actual file)
        weights_path = directory / _WEIGHTS_FILENAME
        weights_dict: dict[str, np.ndarray] = {}
        for i, (weight, bias) in enumerate(self.parameters):
            weights_dict[f"weight_{i}"] = weight
            weights_dict[f"bias_{i}"] = bias
        np.savez_compressed(weights_path, **weights_dict)  # type: ignore[arg-type]

        # Compute checksum from the actual saved file
        with open(weights_path, "rb") as f:
            file_bytes = f.read()
        weights_checksum = hashlib.sha256(file_bytes).hexdigest()

        # Create updated metadata with correct checksum
        updated_metadata = ArtifactMetadata(
            schema_version=self.metadata.schema_version,
            artifact_id=self.metadata.artifact_id,
            architecture=self.metadata.architecture,
            train_config=self.metadata.train_config,
            seed=self.metadata.seed,
            dataset=self.metadata.dataset,
            history=self.metadata.history,
            best_epoch=self.metadata.best_epoch,
            best_val_metrics=self.metadata.best_val_metrics,
            test_metrics=self.metadata.test_metrics,
            weights_checksum=weights_checksum,
            created_at=self.metadata.created_at,
            description=self.metadata.description,
        )

        # Save metadata
        metadata_path = directory / _METADATA_FILENAME
        metadata_path.write_text(updated_metadata.to_json(), encoding="utf-8")

        # Update internal metadata to match saved state
        object.__setattr__(self, "metadata", updated_metadata)

    @classmethod
    def load(cls, directory: Path) -> ModelArtifact:
        """
        Load and validate a ModelArtifact from a directory.

        Validates:
        - Schema version compatibility
        - Required metadata fields present
        - Parameter shapes match architecture
        - All parameter values are finite
        - Weights checksum matches
        """
        if not directory.is_dir():
            raise ArtifactError(f"Artifact directory does not exist: {directory}")

        metadata_path = directory / _METADATA_FILENAME
        weights_path = directory / _WEIGHTS_FILENAME

        if not metadata_path.is_file():
            raise ArtifactError(f"Missing metadata.json: {metadata_path}")
        if not weights_path.is_file():
            raise ArtifactError(f"Missing weights.npz: {weights_path}")

        # Load and parse metadata
        try:
            metadata = ArtifactMetadata.from_json(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ArtifactError(f"Invalid metadata.json: {exc}") from exc

        # Validate schema version
        if metadata.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactError(
                f"Unsupported schema version {metadata.schema_version}; "
                f"current is {ARTIFACT_SCHEMA_VERSION}"
            )

        # Load weights
        try:
            npz = np.load(weights_path)
        except Exception as exc:
            raise ArtifactError(f"Failed to load weights.npz: {exc}") from exc

        # Reconstruct parameters in order
        parameters = []
        expected_depth = metadata.architecture.depth
        for i in range(expected_depth):
            weight_key = f"weight_{i}"
            bias_key = f"bias_{i}"
            if weight_key not in npz or bias_key not in npz:
                raise ArtifactError(f"Missing layer {i} in weights.npz")
            weight = npz[weight_key]
            bias = npz[bias_key]
            parameters.append((weight, bias))

        # Validate parameter shapes match architecture
        expected_conv_layers = metadata.architecture.conv_layers
        if len(parameters) != len(expected_conv_layers):
            raise ArtifactError(
                "Parameter layer count "
                f"{len(parameters)} != architecture depth {len(expected_conv_layers)}"
            )

        for i, ((weight, bias), (in_ch, out_ch, kernel)) in enumerate(
            zip(parameters, expected_conv_layers, strict=True)
        ):
            if weight.shape != (kernel, kernel, in_ch, out_ch):
                expected = (kernel, kernel, in_ch, out_ch)
                raise ArtifactError(f"Layer {i} weight shape {weight.shape} != expected {expected}")
            if bias.shape != (out_ch,):
                raise ArtifactError(f"Layer {i} bias shape {bias.shape} != expected {(out_ch,)}")

        # Validate all values are finite
        for i, (weight, bias) in enumerate(parameters):
            if not np.all(np.isfinite(weight)):
                raise ArtifactError(f"Layer {i} weight contains non-finite values")
            if not np.all(np.isfinite(bias)):
                raise ArtifactError(f"Layer {i} bias contains non-finite values")

        # Verify checksum
        artifact = cls(metadata=metadata, parameters=tuple(parameters))
        artifact._verify_checksum(weights_path)

        return artifact

    def _verify_checksum(self, weights_path: Path) -> None:
        """Verify the weights file checksum matches metadata."""
        with open(weights_path, "rb") as f:
            file_bytes = f.read()
        file_checksum = hashlib.sha256(file_bytes).hexdigest()
        if file_checksum != self.metadata.weights_checksum:
            raise ArtifactError(
                f"Checksum mismatch: expected {self.metadata.weights_checksum}, got {file_checksum}"
            )

    def reconstruct_model(self) -> SuitabilityCNN:
        """
        Reconstruct the SuitabilityCNN from this artifact.

        Returns a new model instance with the saved parameters loaded.
        """
        # The seed is only used for initialization; parameters will be overwritten.
        model = SuitabilityCNN(seed=self.metadata.seed)
        expected_params = model.parameters()
        if len(self.parameters) != len(expected_params):
            raise ArtifactError(
                f"Artifact has {len(self.parameters)} layers, model has {len(expected_params)}"
            )
        for (weight, bias), (saved_weight, saved_bias) in zip(
            expected_params, self.parameters, strict=True
        ):
            weight[:] = saved_weight
            bias[:] = saved_bias
        return model


def _generate_artifact_id(
    model: SuitabilityCNN,
    train_config: TrainConfig,
    seed: int,
    dataset: Dataset,
) -> str:
    """Generate a deterministic artifact ID from model + config + seed + dataset.

    Uses dataset structure (name, size, split counts, image IDs) rather than
    filesystem path to ensure reproducibility across different locations.
    """
    # Create a hash from key identifying information
    hasher = hashlib.sha256()
    hasher.update(model.seed.to_bytes(8, "little"))
    hasher.update(str(train_config.epochs).encode())
    hasher.update(str(train_config.batch_size).encode())
    hasher.update(str(train_config.learning_rate).encode())
    hasher.update(str(train_config.beta1).encode())
    hasher.update(str(train_config.beta2).encode())
    hasher.update(str(train_config.eps).encode())
    hasher.update(str(seed).encode())
    hasher.update(dataset.name.encode())
    hasher.update(str(dataset.size).encode())
    # Include sorted image IDs for content-based identity
    for info in sorted(dataset.images, key=lambda i: i.image_id):
        hasher.update(info.image_id.encode())
    # Use first 16 hex chars (64 bits) for a reasonably short but unique ID
    return hasher.hexdigest()[:16]


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
