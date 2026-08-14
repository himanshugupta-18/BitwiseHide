"""
Training entry point for the BitwiseHide suitability CNN (Phase 2.8.4).

This module provides a reproducible, deterministic training pipeline that uses:
- Phase 2.8.1 dataset discovery (ai.prepare_dataset)
- Phase 2.8.1 deterministic splitting (ai.split)
- Phase 2.8.2 label generation (ai.labels)
- Phase 2.8.3 CNN baseline (ai.cnn)
- Phase 2.8.3 dataloader (ai.dataloader)
- Phase 2.8.3 training infrastructure (ai.training)

Two modes:
- Synthetic smoke training: fully offline, no network, no BSDS500 download
- Real dataset training: requires a local BSDS500 dataset root

Best-model selection is based ONLY on validation performance. Test metrics are
reported for the selected model but NEVER used for model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai.artifact import ModelArtifact
from ai.cnn import SuitabilityCNN
from ai.dataloader import SuitabilityDataset
from ai.prepare_dataset import Dataset, discover_bsds500, write_synthetic_dataset
from ai.split import Split, resolve_split
from ai.training import (
    SplitMetrics,
    TrainConfig,
    TrainingHistory,
    evaluate,
    restore_parameters,
    snapshot_parameters,
    train,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class RunConfig:
    """Immutable configuration for a complete training run.

    Attributes:
        seed: Global seed controlling model initialization, dataset split
            fallback (if needed), and training batch shuffling.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        batch_size: Training batch size.
        beta1: Adam beta1.
        beta2: Adam beta2.
        eps: Adam epsilon.
        training_size: Optional fixed spatial size (width, height) to resample
            all inputs to. Required for mixed-size datasets like BSDS500.
            Omitted for uniform-size datasets (e.g., synthetic smoke).
    """

    seed: int = 0
    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 8
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    training_size: tuple[int, int] | None = None

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
        if self.training_size is not None and (
            len(self.training_size) != 2 or self.training_size[0] < 1 or self.training_size[1] < 1
        ):
            msg = (
                f"training_size must be (width, height) with positive ints, "
                f"got {self.training_size}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class RunResult:
    """Complete result of a training run.

    Attributes:
        history: Per-epoch training and validation MSE.
        best_epoch: Zero-based index of the epoch with the lowest validation MSE.
        best_val_mse: Validation MSE at the best epoch.
        best_val_mae: Validation MAE at the best epoch.
        test_mse: Test MSE of the best-validation model.
        test_mae: Test MAE of the best-validation model.
        test_count: Number of test images evaluated.
    """

    history: TrainingHistory
    best_epoch: int
    best_val_mse: float
    best_val_mae: float
    test_mse: float
    test_mae: float
    test_count: int


def _build_datasets(
    dataset: Dataset,
    *,
    training_size: tuple[int, int] | None,
) -> tuple[SuitabilityDataset, SuitabilityDataset, SuitabilityDataset]:
    """Build train/validation/test SuitabilityDataset from a resolved split."""
    splits = resolve_split(dataset)
    train_ds = SuitabilityDataset(splits[Split.TRAIN], training_size=training_size)
    val_ds = SuitabilityDataset(splits[Split.VAL], training_size=training_size)
    test_ds = SuitabilityDataset(splits[Split.TEST], training_size=training_size)
    return train_ds, val_ds, test_ds


def _create_train_config(run_config: RunConfig) -> TrainConfig:
    """Create TrainConfig from RunConfig."""
    return TrainConfig(
        epochs=run_config.epochs,
        batch_size=run_config.batch_size,
        learning_rate=run_config.learning_rate,
        beta1=run_config.beta1,
        beta2=run_config.beta2,
        eps=run_config.eps,
        seed=run_config.seed,
    )


def run_synthetic_training(
    *,
    per_split: int = 2,
    size: tuple[int, int] = (64, 64),
    run_config: RunConfig | None = None,
    tmp_root: Path | None = None,
) -> RunResult:
    """Run a complete synthetic smoke training end-to-end.

    This is the offline, no-network, no-BSDS500 path for tests and CI.
    It exercises: synthetic dataset -> split -> dataloader -> CNN -> training
    -> evaluation -> best-model selection.

    Args:
        per_split: Number of synthetic images per train/val/test split.
        size: Spatial size of every generated image (width, height).
        run_config: Training hyperparameters; defaults used if omitted.
        tmp_root: Directory to write the synthetic dataset into. If None,
            a temporary directory is created under the system temp.

    Returns:
        Complete RunResult with history, best-model metrics, and test metrics.
    """
    if run_config is None:
        run_config = RunConfig()

    import tempfile

    if tmp_root is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="bitwisehide_synthetic_"))
    else:
        tmp_dir = tmp_root
        tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write synthetic dataset with official split layout
        dataset = write_synthetic_dataset(
            tmp_dir / "synthetic",
            per_split=per_split,
            size=size,
            seed=run_config.seed,
        )

        # Build datasets
        train_ds, val_ds, test_ds = _build_datasets(dataset, training_size=run_config.training_size)

        # Train with best-by-validation checkpointing
        return _run_training_with_checkpointing(train_ds, val_ds, test_ds, run_config)

    finally:
        # Clean up if we created a temporary directory
        if tmp_root is None:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_real_training(
    dataset_root: Path,
    *,
    run_config: RunConfig | None = None,
) -> RunResult:
    """Run training on a local BSDS500 dataset.

    The dataset must already exist locally; no download is attempted.
    The dataset root is passed to ai.prepare_dataset.discover_bsds500.

    Args:
        dataset_root: Local path to a BSDS500-style dataset directory.
        run_config: Training hyperparameters; defaults used if omitted.

    Returns:
        Complete RunResult with history, best-model metrics, and test metrics.

    Raises:
        DatasetError: If the dataset root is invalid, missing, or fails validation.
    """
    if run_config is None:
        run_config = RunConfig()

    # Discover and validate the dataset
    dataset = discover_bsds500(dataset_root)

    # BSDS500 has mixed image sizes, so training_size is REQUIRED
    if run_config.training_size is None:
        raise ValueError(
            "Real BSDS500 datasets have mixed image sizes; "
            "run_config.training_size (width, height) must be set."
        )

    # Build datasets
    train_ds, val_ds, test_ds = _build_datasets(dataset, training_size=run_config.training_size)

    # Train with best-by-validation checkpointing
    return _run_training_with_checkpointing(train_ds, val_ds, test_ds, run_config)


def _run_training_with_checkpointing(
    train_ds: SuitabilityDataset,
    val_ds: SuitabilityDataset,
    test_ds: SuitabilityDataset,
    run_config: RunConfig,
) -> RunResult:
    """Core training loop with best-by-validation model selection.

    Creates the model, trains with epoch callbacks that snapshot the
    validation-best state, restores the best model, and evaluates on test.
    """
    _run = _run_training_with_checkpointing_internal(train_ds, val_ds, test_ds, run_config)
    return _run.result


def _run_training_with_checkpointing_internal(
    train_ds: SuitabilityDataset,
    val_ds: SuitabilityDataset,
    test_ds: SuitabilityDataset,
    run_config: RunConfig,
) -> _TrainingRun:
    """Internal training that returns model and dataset for artifact creation."""
    train_config = _create_train_config(run_config)

    # Initialize model with the global seed for reproducibility
    model = SuitabilityCNN(seed=run_config.seed)

    # Track best validation state
    best_val_mse = float("inf")
    best_snapshot: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
    best_epoch = -1

    def epoch_callback(epoch: int, current_model: SuitabilityCNN) -> None:
        nonlocal best_val_mse, best_snapshot, best_epoch
        val_metrics = evaluate(current_model, val_ds)
        if val_metrics.mse < best_val_mse:
            best_val_mse = val_metrics.mse
            best_snapshot = snapshot_parameters(current_model)
            best_epoch = epoch

    # Train with the callback
    history = train(
        model,
        train_ds,
        config=train_config,
        val_dataset=val_ds,
        epoch_callback=epoch_callback,
    )

    # Restore the best validation model
    if best_snapshot is None:
        raise RuntimeError("No best model was selected (validation never ran).")
    restore_parameters(model, best_snapshot)

    # Evaluate best model on validation and test
    best_val_metrics = evaluate(model, val_ds)
    test_metrics = evaluate(model, test_ds)

    result = RunResult(
        history=history,
        best_epoch=best_epoch,
        best_val_mse=best_val_metrics.mse,
        best_val_mae=best_val_metrics.mae,
        test_mse=test_metrics.mse,
        test_mae=test_metrics.mae,
        test_count=test_metrics.count,
    )

    return _TrainingRun(
        result=result, model=model, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds
    )


@dataclass(frozen=True)
class _TrainingRun:
    """Internal container for training run with model and datasets."""

    result: RunResult
    model: SuitabilityCNN
    train_ds: SuitabilityDataset
    val_ds: SuitabilityDataset
    test_ds: SuitabilityDataset


def create_artifact(
    result: RunResult,
    model: SuitabilityCNN,
    *,
    run_config: RunConfig,
    dataset: Dataset,
    artifact_id: str | None = None,
    description: str = "",
) -> ModelArtifact:
    """
    Create a ModelArtifact from a training run result.

    This extracts the best-validation model (already restored in `model`)
    and packages it with all metadata needed for reproducibility.

    Args:
        result: The RunResult from the training run.
        model: The trained model (must be at the best epoch state).
        run_config: The RunConfig used for training.
        dataset: The dataset used for training.
        artifact_id: Optional explicit artifact ID; generated if omitted.
        description: Optional human-readable description.

    Returns:
        A complete ModelArtifact ready to save or register.
    """
    best_val_metrics = SplitMetrics(
        mse=result.best_val_mse,
        mae=result.best_val_mae,
        count=len(dataset.images) // 3,  # Approximate validation split count
    )
    test_metrics = SplitMetrics(
        mse=result.test_mse,
        mae=result.test_mae,
        count=result.test_count,
    )
    return ModelArtifact.create(
        model=model,
        train_config=_create_train_config(run_config),
        seed=run_config.seed,
        dataset=dataset,
        history=result.history,
        best_epoch=result.best_epoch,
        best_val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        artifact_id=artifact_id,
        description=description,
    )
