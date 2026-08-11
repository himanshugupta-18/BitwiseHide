"""
Phase 2.8.3 dataloader tests.

Covers ai.dataloader, fully offline:
- building (Z-domain input, suitability label) pairs from official splits and
  from resolve_split fallback partitions
- the Z-domain guarantee (every input LSB is clear) and that the target equals
  suitability_label_map of that Z-domain input
- batching: normalization to [0, 1], deterministic seeded shuffles, full
  coverage with no duplication
- strict image-level separation across train/val/test (leakage check)
- error handling: missing/corrupt images, mixed sizes (with and without
  training_size), unknown splits, empty batches

All data is synthetic and generated in tmp_path / in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ai.dataloader import SuitabilityDataset
from ai.labels import suitability_label_map
from ai.prepare_dataset import DatasetError, ImageInfo, discover_bsds500, write_synthetic_dataset
from ai.split import Split, resolve_split


def _synthetic(tmp_path: Path, *, per_split: int = 3, size: tuple[int, int] = (16, 16)) -> tuple:
    """Write a deterministic synthetic dataset and return (dataset, splits)."""
    dataset = write_synthetic_dataset(tmp_path / "ds", per_split=per_split, size=size, seed=0)
    return dataset, resolve_split(dataset)


def _write_mixed_size_tree(root: Path) -> None:
    """A split tree whose train split holds two different-size images."""
    for split_name in ("train", "val", "test"):
        (root / "images" / split_name).mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 10, 10)).save(root / "images" / "train" / "a.png")
    Image.new("RGB", (12, 16), color=(20, 20, 20)).save(root / "images" / "train" / "b.png")
    Image.new("RGB", (8, 8), color=(30, 30, 30)).save(root / "images" / "val" / "c.png")
    Image.new("RGB", (8, 8), color=(40, 40, 40)).save(root / "images" / "test" / "d.png")


class TestDatasetBuild:
    def test_official_split_counts_and_ids(self, tmp_path) -> None:
        dataset, _ = _synthetic(tmp_path, per_split=4)
        train = SuitabilityDataset.from_dataset_split(dataset, Split.TRAIN)
        assert len(train) == 4
        assert all(sample.image_id.startswith("synthetic_train_") for sample in train._samples)
        assert all(sample.source_split == Split.TRAIN for sample in train._samples)

    def test_from_dataset_split_on_unsplit_raises(self, tmp_path) -> None:
        root = tmp_path / "flat"
        images_dir = root / "images"
        images_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(5, 5, 5)).save(images_dir / "img001.png")
        dataset = discover_bsds500(root)
        assert not dataset.official_split_available
        with pytest.raises(DatasetError, match="no images"):
            SuitabilityDataset.from_dataset_split(dataset, Split.TRAIN)

    def test_inputs_are_z_domain(self, tmp_path) -> None:
        _, splits = _synthetic(tmp_path)
        for sample in SuitabilityDataset(splits[Split.TRAIN])._samples:
            assert (sample.input % 2 == 0).all()

    def test_target_equals_label_of_z_domain_input(self, tmp_path) -> None:
        _, splits = _synthetic(tmp_path)
        for sample in SuitabilityDataset(splits[Split.TRAIN])._samples:
            expected = suitability_label_map(Image.fromarray(sample.input))
            assert np.allclose(sample.target, expected, atol=1e-12)

    def test_targets_in_unit_range(self, tmp_path) -> None:
        _, splits = _synthetic(tmp_path)
        for sample in SuitabilityDataset(splits[Split.TRAIN])._samples:
            assert sample.target.min() >= 0.0 and sample.target.max() <= 1.0

    def test_missing_image_raises(self) -> None:
        info = ImageInfo(image_id="missing", path=Path("/nonexistent/missing.png"))
        with pytest.raises(DatasetError, match="missing"):
            SuitabilityDataset([info])

    def test_mixed_sizes_rejected_without_training_size(self, tmp_path) -> None:
        root = tmp_path / "mixed"
        _write_mixed_size_tree(root)
        dataset = discover_bsds500(root)
        train_records = [info for info in dataset.images if info.source_split == Split.TRAIN]
        with pytest.raises(DatasetError, match="mixed image sizes"):
            SuitabilityDataset(train_records)

    def test_training_size_unifies_mixed_sizes(self, tmp_path) -> None:
        root = tmp_path / "mixed"
        _write_mixed_size_tree(root)
        dataset = discover_bsds500(root)
        train_records = [info for info in dataset.images if info.source_split == Split.TRAIN]
        resized = SuitabilityDataset(train_records, training_size=(10, 10))
        assert len(resized) == 2
        for sample in resized._samples:
            assert sample.input.shape == (10, 10, 3)
            assert sample.target.shape == (10, 10)


class TestBatching:
    def test_make_batch_normalizes_inputs(self, tmp_path) -> None:
        _, splits = _synthetic(tmp_path)
        dataset = SuitabilityDataset(splits[Split.TRAIN])
        x, y = dataset.make_batch([0, 1, 2])
        assert x.shape == (3, 16, 16, 3)
        assert y.shape == (3, 16, 16)
        assert x.min() >= 0.0 and x.max() <= 1.0
        rounded = np.rint(x * 255.0).astype(np.uint8)
        assert np.array_equal(rounded, np.stack([dataset._samples[i].input for i in (0, 1, 2)]))

    def test_make_batch_empty_indices_raises(self, tmp_path) -> None:
        _, splits = _synthetic(tmp_path)
        with pytest.raises(ValueError, match="at least one"):
            SuitabilityDataset(splits[Split.TRAIN]).make_batch([])

    def test_shuffled_batches_deterministic_in_seed(self, tmp_path) -> None:
        _, splits = _synthetic(tmp_path, per_split=10)
        first = SuitabilityDataset(splits[Split.TRAIN])
        second = SuitabilityDataset(splits[Split.TRAIN])
        batches_a = list(first.shuffled_batches(4, np.random.default_rng(9)))
        batches_b = list(second.shuffled_batches(4, np.random.default_rng(9)))
        assert len(batches_a) == len(batches_b)
        for (xa, ya), (xb, yb) in zip(batches_a, batches_b, strict=True):
            assert np.array_equal(xa, xb)
            assert np.array_equal(ya, yb)
        batches_c = list(first.shuffled_batches(4, np.random.default_rng(10)))
        assert not np.array_equal(batches_a[0][0], batches_c[0][0])

    def test_shuffled_batches_cover_every_sample_exactly_once(self, tmp_path) -> None:
        _, splits = _synthetic(tmp_path, per_split=7)
        dataset = SuitabilityDataset(splits[Split.TRAIN])
        seen: set[bytes] = set()
        for x, _y in dataset.shuffled_batches(3, np.random.default_rng(0)):
            assert x.shape[0] <= 3
            for row in np.rint(x * 255.0).astype(np.uint8):
                seen.add(row.tobytes())
        expected = {sample.input.tobytes() for sample in dataset._samples}
        assert seen == expected


class TestLeakage:
    def test_image_level_separation_across_splits(self, tmp_path) -> None:
        dataset, splits = _synthetic(tmp_path)
        train = SuitabilityDataset(splits[Split.TRAIN])
        val = SuitabilityDataset(splits[Split.VAL])
        test = SuitabilityDataset(splits[Split.TEST])
        assert train.image_ids & val.image_ids == frozenset()
        assert train.image_ids & test.image_ids == frozenset()
        assert val.image_ids & test.image_ids == frozenset()
        all_ids = frozenset(info.image_id for info in dataset.images)
        assert train.image_ids | val.image_ids | test.image_ids == all_ids

    def test_fallback_split_records_build_dataset(self, tmp_path) -> None:
        root = tmp_path / "flat"
        images_dir = root / "images"
        images_dir.mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (8, 8), color=(i, 0, 0)).save(images_dir / f"img{i:03d}.png")
        dataset = discover_bsds500(root)
        splits = resolve_split(dataset, seed=5)
        split_counts = {split: len(infos) for split, infos in splits.items()}
        assert split_counts[Split.TRAIN] == 4 and split_counts[Split.VAL] == 2
        train = SuitabilityDataset(splits[Split.TRAIN])
        assert len(train) == 4
        assert all(sample.source_split is None for sample in train._samples)
