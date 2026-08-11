"""
Phase 2.8.1 dataset splitting tests.

Covers ai.split, fully offline:
- deterministic and ordering-independent image-level splitting
- exact coverage (every image in exactly one split) and ratio adherence
- leakage checks: duplicate image_ids rejected, assert_no_leakage catches a
  cross-split duplicate
- resolve_split preferring the official BSDS500 split and falling back
  deterministically on unsplit datasets

All datasets are synthetic and generated in memory / in tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ai.prepare_dataset import (
    DatasetError,
    ImageInfo,
    Split,
    discover_bsds500,
    write_synthetic_dataset,
)
from ai.split import (
    DEFAULT_RATIOS,
    assert_no_leakage,
    resolve_split,
    split_counts,
    split_dataset,
)


def _infos(*image_ids: str) -> list[ImageInfo]:
    """Build ImageInfo records with synthetic image paths (never opened on disk)."""
    return [ImageInfo(image_id=image_id, path=Path(f"{image_id}.png")) for image_id in image_ids]


class TestSplitDataset:
    def test_deterministic(self) -> None:
        images = _infos(*(f"img{i:04d}" for i in range(500)))
        first = split_dataset(images)
        second = split_dataset(images)
        assert first == second

    def test_ordering_independent(self) -> None:
        images = _infos(*(f"img{i:03d}" for i in range(50)))
        shuffled = list(images)
        np.random.default_rng(0).shuffle(shuffled)
        assert split_dataset(images, seed=9) == split_dataset(shuffled, seed=9)

    def test_exact_coverage_and_no_overlap(self) -> None:
        images = _infos(*(f"img{i:03d}" for i in range(100)))
        splits = split_dataset(images, ratios=(0.4, 0.2, 0.4), seed=1)
        assert split_counts(splits) == {Split.TRAIN: 40, Split.VAL: 20, Split.TEST: 40}
        flat = [info for infos in splits.values() for info in infos]
        assert len(flat) == len(images)  # no duplication
        assert sorted(info.image_id for info in flat) == sorted(info.image_id for info in images)

    def test_ratios_defaults_to_bsds500_proportions(self) -> None:
        assert DEFAULT_RATIOS == (0.4, 0.2, 0.4)

    def test_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_dataset(_infos("a", "b"), ratios=(0.5, 0.5, 0.2))

    def test_ratios_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            split_dataset(_infos("a", "b"), ratios=(1.0, 0.0, 0.0))

    def test_duplicate_image_id_rejected(self) -> None:
        images = [
            ImageInfo("dup", Path("/a/x.png")),
            ImageInfo("dup", Path("/b/y.png")),
        ]
        with pytest.raises(DatasetError, match="duplicate image_id"):
            split_dataset(images)

    def test_every_image_in_exactly_one_split(self) -> None:
        images = _infos(*(f"img{i:02d}" for i in range(25)))
        splits = split_dataset(images, seed=2)
        seen: dict[str, int] = {}
        for infos in splits.values():
            for info in infos:
                seen[info.image_id] = seen.get(info.image_id, 0) + 1
        assert set(seen) == {info.image_id for info in images}
        assert set(seen.values()) == {1}


class TestLeakage:
    def test_assert_no_leakage_passes_for_disjoint(self) -> None:
        splits = {
            Split.TRAIN: _infos("a", "b"),
            Split.VAL: _infos("c"),
            Split.TEST: _infos("d", "e"),
        }
        assert_no_leakage(splits)  # does not raise

    def test_assert_no_leakage_rejects_cross_split_duplicate(self) -> None:
        splits = {
            Split.TRAIN: _infos("a", "b"),
            Split.VAL: _infos("a", "c"),
            Split.TEST: _infos("d"),
        }
        with pytest.raises(DatasetError, match="leakage"):
            assert_no_leakage(splits)


class TestResolveSplit:
    def test_prefers_official_split(self, tmp_path) -> None:
        dataset = write_synthetic_dataset(tmp_path / "ds", per_split=3, size=(16, 16), seed=0)
        assert dataset.official_split_available
        splits = resolve_split(dataset)
        assert split_counts(splits) == {Split.TRAIN: 3, Split.VAL: 3, Split.TEST: 3}
        # The official layout's folder tag is preserved.
        assert all(info.source_split == Split.TRAIN for info in splits[Split.TRAIN])
        assert all(info.source_split == Split.VAL for info in splits[Split.VAL])

    def test_falls_back_deterministically(self, tmp_path) -> None:
        root = tmp_path / "flat"
        images_dir = root / "images"
        images_dir.mkdir(parents=True)
        for i in range(20):
            Image.new("RGB", (8, 8), color=(i, 0, 0)).save(images_dir / f"img{i:03d}.png")
        dataset = discover_bsds500(root)
        assert not dataset.official_split_available
        first = resolve_split(dataset, seed=5)
        second = resolve_split(dataset, seed=5)
        assert first == second
        assert split_counts(first) == {Split.TRAIN: 8, Split.VAL: 4, Split.TEST: 8}
