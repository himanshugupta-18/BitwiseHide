"""
Phase 2.8.1 dataset preparation tests.

Covers ai.prepare_dataset, fully offline:
- missing / empty / corrupt dataset handling (DatasetError)
- BSDS500-style official split discovery: split folders and iids id lists
- flat unsplit discovery
- duplicate-image_id rejection and incomplete id-list rejection (leakage guards)
- write_synthetic_dataset determinism and round-trip through discovery
- load_image_rgb RGB normalization and missing-file handling

All images are generated in memory or written to pytest's tmp_path.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ai.prepare_dataset import (
    BSDS500_EXPECTED_COUNTS,
    DatasetError,
    Split,
    discover_bsds500,
    load_image_rgb,
    verify_bsds500,
    write_synthetic_dataset,
)


def _write_png(directory, name, *, color=(1, 2, 3), mode="RGB", size=(8, 8)):
    """Write a small PIL image as PNG into `directory` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new(mode, size, color=color).save(path, format="PNG")
    return path


class TestMissingAndInvalid:
    def test_missing_root_raises(self, tmp_path) -> None:
        with pytest.raises(DatasetError, match="does not exist"):
            discover_bsds500(tmp_path / "not-there")

    def test_empty_root_raises(self, tmp_path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        with pytest.raises(DatasetError, match="No images found"):
            discover_bsds500(root)

    def test_corrupt_image_raises(self, tmp_path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "bad.jpg").write_bytes(b"this is not a jpeg")
        with pytest.raises(DatasetError, match="failed RGB validation"):
            discover_bsds500(tmp_path)

    def test_partial_split_dirs_raise(self, tmp_path) -> None:
        root = tmp_path / "partial"
        (root / "train").mkdir(parents=True)
        (root / "val").mkdir()
        _write_png(root / "train", "a.png")
        _write_png(root / "val", "b.png")
        with pytest.raises(DatasetError, match="Partial split directories"):
            discover_bsds500(root)

    def test_non_image_extension_ignored(self, tmp_path) -> None:
        root = tmp_path / "mixed"
        (root / "images").mkdir(parents=True)
        (root / "images" / "notes.txt").write_text("not an image")
        _write_png(root / "images", "real.png")
        dataset = discover_bsds500(root)
        assert dataset.size == 1
        assert dataset.images[0].image_id == "real"


class TestOfficialSplit:
    def test_split_folders_discovered(self, tmp_path) -> None:
        root = tmp_path / "splitfolders"
        for split_name in ("train", "val", "test"):
            _write_png(root / split_name, f"img_{split_name}.png")
        dataset = discover_bsds500(root)
        assert dataset.name == "bsds500"
        assert dataset.official_split_available
        assert dataset.size == 3
        assert dataset.counts_by_split() == {Split.TRAIN: 1, Split.VAL: 1, Split.TEST: 1}
        by_id = {info.image_id: info for info in dataset.images}
        assert by_id["img_train"].source_split == Split.TRAIN
        assert by_id["img_test"].source_split == Split.TEST

    def test_images_subdir_variant(self, tmp_path) -> None:
        root = tmp_path / "imagesdir"
        _write_png(root / "images" / "train", "a.png")
        _write_png(root / "images" / "val", "b.png")
        _write_png(root / "images" / "test", "c.png")
        dataset = discover_bsds500(root)
        assert dataset.official_split_available
        assert dataset.size == 3

    def test_id_lists_define_split(self, tmp_path) -> None:
        root = tmp_path / "idlists"
        images_dir = root / "images"
        for name in ("a", "b", "c", "d", "e"):
            _write_png(images_dir, f"{name}.png")
        (images_dir / "iids_train.txt").write_text("a\nb\n")
        (images_dir / "iids_val.txt").write_text("c\n")
        (images_dir / "iids_test.txt").write_text("d\ne\n")
        dataset = discover_bsds500(root)
        assert dataset.official_split_available
        assert dataset.counts_by_split() == {Split.TRAIN: 2, Split.VAL: 1, Split.TEST: 2}
        assert all(info.source_split is not None for info in dataset.images)

    def test_id_list_accepts_jpg_suffix(self, tmp_path) -> None:
        root = tmp_path / "idsuffix"
        images_dir = root / "images"
        _write_png(images_dir, "100007.png")
        _write_png(images_dir, "100008.png")
        (images_dir / "iids_train.txt").write_text("100007.jpg\n")
        (images_dir / "iids_val.txt").write_text("100008.jpg\n")
        (images_dir / "iids_test.txt").write_text("")
        dataset = discover_bsds500(root)
        assert dataset.counts_by_split()[Split.TRAIN] == 1
        assert dataset.counts_by_split()[Split.TEST] == 0

    def test_incomplete_id_lists_raise(self, tmp_path) -> None:
        root = tmp_path / "incomplete"
        images_dir = root / "images"
        for name in ("a", "b", "c"):
            _write_png(images_dir, f"{name}.png")
        (images_dir / "iids_train.txt").write_text("a\n")
        (images_dir / "iids_val.txt").write_text("b\n")
        (images_dir / "iids_test.txt").write_text("c\n")
        _write_png(images_dir, "orphan.png")
        with pytest.raises(DatasetError, match="not covered by the iids"):
            discover_bsds500(root)

    def test_id_in_duplicate_lists_raise(self, tmp_path) -> None:
        root = tmp_path / "dupids"
        images_dir = root / "images"
        _write_png(images_dir, "a.png")
        (images_dir / "iids_train.txt").write_text("a\n")
        (images_dir / "iids_val.txt").write_text("a\n")
        (images_dir / "iids_test.txt").write_text("")
        with pytest.raises(DatasetError, match="multiple iids files"):
            discover_bsds500(root)


class TestDuplicates:
    def test_duplicate_image_id_across_splits_raises(self, tmp_path) -> None:
        root = tmp_path / "dupstem"
        for split_name in ("train", "val", "test"):
            split_dir = root / split_name
            split_dir.mkdir(parents=True)
        _write_png(root / "train", "x.png")
        _write_png(root / "val", "x.png")
        _write_png(root / "test", "y.png")
        with pytest.raises(DatasetError, match="duplicate image_id"):
            discover_bsds500(root)


class TestFlat:
    def test_flat_unsplit(self, tmp_path) -> None:
        root = tmp_path / "flat"
        images_dir = root / "images"
        for name in ("a", "b", "c"):
            _write_png(images_dir, f"{name}.png")
        dataset = discover_bsds500(root)
        assert not dataset.official_split_available
        assert dataset.size == 3
        assert all(info.source_split is None for info in dataset.images)


class TestWriteSynthetic:
    def test_round_trips_as_official_split(self, tmp_path) -> None:
        dataset = write_synthetic_dataset(tmp_path / "ds", per_split=3, size=(32, 32), seed=1)
        assert dataset.name == "synthetic"
        assert dataset.official_split_available
        assert dataset.size == 9
        assert dataset.counts_by_split() == {Split.TRAIN: 3, Split.VAL: 3, Split.TEST: 3}
        assert all(info.source_split is not None for info in dataset.images)

    def test_deterministic_across_roots(self, tmp_path) -> None:
        root1 = tmp_path / "a"
        root2 = tmp_path / "b"
        dataset1 = write_synthetic_dataset(root1, per_split=2, size=(16, 16), seed=7)
        dataset2 = write_synthetic_dataset(root2, per_split=2, size=(16, 16), seed=7)
        assert dataset1.size == dataset2.size == 6
        files = sorted(str(path.relative_to(root1)) for path in root1.rglob("*.png"))
        files2 = sorted(str(path.relative_to(root2)) for path in root2.rglob("*.png"))
        assert files == files2
        for relative in files:
            arr1 = np.asarray(Image.open(root1 / relative), dtype=np.uint8)
            arr2 = np.asarray(Image.open(root2 / relative), dtype=np.uint8)
            assert np.array_equal(arr1, arr2)

    def test_different_seed_differs(self, tmp_path) -> None:
        root1 = tmp_path / "s1"
        root2 = tmp_path / "s2"
        write_synthetic_dataset(root1, per_split=1, size=(16, 16), seed=0)
        write_synthetic_dataset(root2, per_split=1, size=(16, 16), seed=1)
        paths1 = sorted(str(path.relative_to(root1)) for path in root1.rglob("*.png"))
        paths2 = sorted(str(path.relative_to(root2)) for path in root2.rglob("*.png"))
        assert paths1 == paths2  # same filenames
        differing = [
            relative
            for relative in paths1
            if not np.array_equal(
                np.asarray(Image.open(root1 / relative), dtype=np.uint8),
                np.asarray(Image.open(root2 / relative), dtype=np.uint8),
            )
        ]
        assert differing, "different seeds should change at least one generated image"


class TestLoadImage:
    def test_load_normalizes_grayscale_to_rgb(self, tmp_path) -> None:
        path = _write_png(tmp_path, "gray.png", color=100, mode="L")
        image = load_image_rgb(path)
        assert image.mode == "RGB"
        arr = np.asarray(image, dtype=np.uint8)
        assert arr.shape == (8, 8, 3)
        assert np.array_equal(arr[0, 0], (100, 100, 100))

    def test_load_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(DatasetError, match="does not exist"):
            load_image_rgb(tmp_path / "nope.png")


class TestVerify:
    def test_verify_counts_pass_when_matching(self, tmp_path) -> None:
        dataset = write_synthetic_dataset(tmp_path / "scaled", per_split=3, size=(16, 16), seed=0)
        expected = {Split.TRAIN: 3, Split.VAL: 3, Split.TEST: 3}
        verify_bsds500(dataset, expected=expected)  # does not raise

    def test_verify_rejects_unsplit(self, tmp_path) -> None:
        root = tmp_path / "flat"
        for name in ("a", "b", "c"):
            _write_png(root / "images", f"{name}.png")
        with pytest.raises(DatasetError, match="unsplit"):
            verify_bsds500(discover_bsds500(root))

    def test_verify_rejects_count_mismatch(self, tmp_path) -> None:
        dataset = write_synthetic_dataset(tmp_path / "small", per_split=1, size=(16, 16), seed=0)
        with pytest.raises(DatasetError, match="do not match"):
            verify_bsds500(dataset, expected=BSDS500_EXPECTED_COUNTS)
