"""
Suitability dataset/dataloader (Phase 2.8.3, training-time only).

Turns the Phase 2.8.1 discovered dataset splits into the exact training pairs
the CNN baseline consumes: the model input is the Z domain (``Z = image & 0xFE``,
see ``ai.cnn.z_domain_array``) and the target is the Phase 2.8.2 label of that
Z-domain image (``ai.labels.suitability_label_map``).

Splits are consumed whole: a ``SuitabilityDataset`` is built from one split's
``ImageInfo`` records (from ``ai.split.resolve_split`` or
``ai.split.split_dataset``), and the caller never mixes records across splits.
The split layer has already guaranteed image-level separation and leakage
freedom, so a dataset's ``image_ids`` are disjoint from every other split's by
construction.

Determinism and sizes:
- Every input is a pure function of the image bytes: Z-domain derivation,
  RGB normalization, and label computation are all deterministic and
  network-free.
- Images in a batch must share one spatial shape. Native-size datasets require
  uniform input sizes (the default, and the case for the deterministic
  synthetic smoke datasets); pass ``training_size`` to center on a fixed grid
  via deterministic resampling so mixed-size corpora (e.g. BSDS500) batch too.

Design constraints:
- Framework-agnostic: no torch or other ML framework is imported.
- Invalid input (a missing/corrupt image) raises ``DatasetError``; invalid
  *arguments* (an empty index list, unknown split) raise ValueError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ai.cnn import z_domain_array
from ai.labels import suitability_label_map
from ai.prepare_dataset import DatasetError, load_image_rgb

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from app.core.evaluation import FloatArray, UInt8Array

    from ai.prepare_dataset import Dataset, ImageInfo, Split


@dataclass(frozen=True)
class SuitabilitySample:
    """One materialized (Z-domain input, suitability label) training pair.

    Attributes:
        image_id: The source image's id (unique across all splits).
        source_split: The split the source image belongs to, or None for an
            unsplit (fallback-partitioned) dataset.
        input: The Z-domain RGB array ``(H, W, 3)`` uint8 — the model input.
        target: The ``(H, W)`` suitability label in [0, 1] computed from the
            Z-domain input.
    """

    image_id: str
    source_split: Split | None
    input: UInt8Array
    target: FloatArray


class SuitabilityDataset:
    """Materialized per-split (Z-domain image, suitability label) dataset.

    Wrap a single split's ``ImageInfo`` records — from ``resolve_split``
    (official or fallback) — and precompute each sample's Z-domain input and
    suitability target. Every sample derives from exactly one source image, so
    image-level separation is inherited from the split.
    """

    def __init__(
        self,
        images: Sequence[ImageInfo],
        *,
        training_size: tuple[int, int] | None = None,
    ) -> None:
        """Build samples from `images`, optionally resizing to `training_size`.

        `training_size` is PIL's ``(width, height)`` convention; inputs are
        resized with LANCZOS and labels with BILINEAR, both deterministically.
        When omitted, all images must already share one spatial size.

        Raises:
            DatasetError: If an image file is missing/corrupt, or the images do
                not share a single spatial size (when `training_size` is None).
        """
        samples: list[SuitabilitySample] = []
        for info in images:
            image = load_image_rgb(info.path)
            z = z_domain_array(image)
            target = suitability_label_map(Image.fromarray(z))
            if training_size is not None:
                z = _resize_input(z, training_size)
                target = _resize_label(target, training_size)
            samples.append(
                SuitabilitySample(
                    image_id=info.image_id,
                    source_split=info.source_split,
                    input=z,
                    target=target,
                )
            )
        self._samples = tuple(samples)
        if training_size is None:
            _assert_uniform_size(self._samples)
        self._image_ids = frozenset(sample.image_id for sample in self._samples)

    @classmethod
    def from_dataset_split(cls, dataset: Dataset, split: Split) -> SuitabilityDataset:
        """Build a dataset from one official split of a discovered `dataset`.

        Only meaningful when ``dataset.official_split_available`` is True (the
        records carry ``source_split``); for unsplit datasets use
        ``resolve_split`` and pass the per-split records directly.

        Raises:
            DatasetError: If the split has no images.
        """
        records = [info for info in dataset.images if info.source_split == split]
        if not records:
            msg = f"Split {split.value!r} has no images in dataset {dataset.name!r}."
            raise DatasetError(msg)
        return cls(records)

    def __len__(self) -> int:
        """Number of samples (source images) in this split."""
        return len(self._samples)

    @property
    def image_ids(self) -> frozenset[str]:
        """The source image ids in this split (unique within the dataset)."""
        return self._image_ids

    def make_batch(self, indices: Sequence[int]) -> tuple[FloatArray, FloatArray]:
        """Stack the given sample indices into a (input, label) batch.

        Inputs are returned as float64 in [0, 1] (uint8 Z-domain / 255.0);
        targets as float64 ``(B, H, W)`` suitability maps in [0, 1].

        Raises:
            ValueError: If `indices` is empty or any index is out of range.
        """
        if not indices:
            msg = "make_batch requires at least one sample index."
            raise ValueError(msg)
        inputs = np.stack([self._samples[index].input for index in indices])
        targets = np.stack([self._samples[index].target for index in indices])
        return (
            np.asarray(inputs, dtype=np.float64) / 255.0,
            np.asarray(targets, dtype=np.float64),
        )

    def shuffled_batches(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Iterator[tuple[FloatArray, FloatArray]]:
        """Yield (input, label) batches in a seeded-random order.

        The permutation is drawn from `rng` (a caller-owned
        ``np.random.Generator``), so the caller controls reproducibility. The
        final batch may be smaller than `batch_size`.
        """
        order = np.arange(len(self), dtype=np.intp)
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            yield self.make_batch(indices.tolist())


# --- Internal helpers ---------------------------------------------------------


def _resize_input(z: UInt8Array, size: tuple[int, int]) -> UInt8Array:
    """Resize a Z-domain RGB array to `size` with LANCZOS (deterministic)."""
    image = Image.fromarray(z, mode="RGB").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _resize_label(label: FloatArray, size: tuple[int, int]) -> FloatArray:
    """Resize a suitability map to `size` with BILINEAR (deterministic)."""
    image = Image.fromarray(np.asarray(label, dtype=np.float32), mode="F")
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64)


def _assert_uniform_size(samples: Sequence[SuitabilitySample]) -> None:
    """Reject datasets whose samples have differing spatial shapes.

    A batch stacks its samples, so all inputs must share one (H, W). Mixed-size
    corpora should pass ``training_size`` to the constructor instead. An empty
    dataset trivially satisfies uniformity.
    """
    if not samples:
        return
    first = samples[0].input.shape[:2]
    for sample in samples[1:]:
        if sample.input.shape[:2] != first:
            msg = (
                "Dataset contains mixed image sizes; pass training_size "
                f"(found {first} and {sample.input.shape[:2]})."
            )
            raise DatasetError(msg)
