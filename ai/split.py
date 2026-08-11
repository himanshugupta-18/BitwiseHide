"""
Deterministic image-level train/validation/test splitting (Phase 2.8.1).

Splits operate on whole source images — never on crops — so a source image can
never straddle two sets. That guarantee is the point: later phases will crop
images for the model, and those crops inherit this image-level separation
because they are all derived from a single split assignment.

Two entry points:
- ``resolve_split`` prefers the dataset's official BSDS500 split (the intended
  200/100/200 layout) whenever it is available, and only falls back to a
  deterministic partition otherwise.
- ``split_dataset`` is the deterministic fallback itself: it sorts records by
  (image_id, path), seeds a fresh RNG with a caller-supplied seed, shuffles, and
  partitions into the requested ratios. Because the input is sorted first, the
  partition is independent of filesystem enumeration order and reproducible.

Leakage prevention
------------------
``assert_no_leakage`` is the explicit check that no image_id occurs in more than
one split. ``split_dataset`` calls it before returning, and ``resolve_split``
calls it when adopting the official layout (so a mis-packaged download with the
same image under two folders fails loudly). Discovery itself also rejects
duplicate image IDs (``prepare_dataset._validate_records``).

Error contract mirrors ``prepare_dataset``: invalid *split state* raises
``DatasetError``; invalid *arguments* (bad ratios) raise ``ValueError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ai.prepare_dataset import Dataset, DatasetError, ImageInfo, Split

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Fallback split ratios, matching BSDS500's official 200/100/200 proportions.
DEFAULT_RATIOS: tuple[float, float, float] = (0.4, 0.2, 0.4)
#: Default seed for the deterministic fallback partition.
DEFAULT_SEED = 42


def split_dataset(
    images: Sequence[ImageInfo],
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[Split, list[ImageInfo]]:
    """Deterministic, image-level split of `images` into train/val/test.

    Records are sorted by (image_id, path) before a seeded shuffle, so the
    result depends only on the records themselves and `seed` — not on input
    order. Every image appears in exactly one split; ``assert_no_leakage`` is
    run before returning.

    Args:
        images: Source image records. Each image_id must be unique.
        ratios: (train, val, test) proportions summing to 1.0. The val split
            absorbs any rounding remainder.
        seed: RNG seed; the same seed always reproduces the same partition.

    Returns:
        A mapping Split -> its image records.

    Raises:
        DatasetError: If any image_id is duplicated in `images`, or (defensively)
            leakage is detected.
        ValueError: If ratios are non-positive or do not sum to 1.0.
    """
    _validate_ratios(ratios)
    records = sorted(images, key=lambda info: (info.image_id, str(info.path)))
    _assert_unique_image_ids(records)

    shuffled = list(records)
    np.random.default_rng(seed).shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * ratios[0])
    n_test = int(total * ratios[2])
    n_val = total - n_train - n_test

    splits = {
        Split.TRAIN: shuffled[:n_train],
        Split.VAL: shuffled[n_train : n_train + n_val],
        Split.TEST: shuffled[n_train + n_val :],
    }
    assert_no_leakage(splits)
    return splits


def resolve_split(
    dataset: Dataset,
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[Split, list[ImageInfo]]:
    """Resolve `dataset` to a split, preferring its official layout.

    When ``dataset.official_split_available`` is True, the intended BSDS500
    split (from train/val/test folders or id lists) is adopted and checked for
    leakage. Otherwise a deterministic fallback partition with `ratios`/`seed`
    is produced.

    Returns:
        A mapping Split -> its image records, guaranteed leakage-free.
    """
    if dataset.official_split_available:
        by_split: dict[Split, list[ImageInfo]] = {split: [] for split in Split}
        for info in dataset.images:
            if info.source_split is None:
                msg = f"Official split advertised but image {info.image_id!r} has no split."
                raise DatasetError(msg)
            by_split[info.source_split].append(info)
        assert_no_leakage(by_split)
        return by_split
    return split_dataset(dataset.images, ratios=ratios, seed=seed)


def split_counts(splits: Mapping[Split, Sequence[ImageInfo]]) -> dict[Split, int]:
    """Per-split image counts for a partition."""
    return {split: len(infos) for split, infos in splits.items()}


def assert_no_leakage(splits: Mapping[Split, Sequence[ImageInfo]]) -> None:
    """Verify that no image_id occurs in more than one split.

    Raises:
        DatasetError: If any image_id appears in two different splits.
    """
    seen: dict[str, Split] = {}
    offenders: list[tuple[str, Split, Split]] = []
    for split, infos in splits.items():
        for info in infos:
            if info.image_id in seen:
                offenders.append((info.image_id, seen[info.image_id], split))
            else:
                seen[info.image_id] = split
    if offenders:
        pairs = (f"{image_id!r} in {first}/{second}" for image_id, first, second in offenders[:5])
        shown = ", ".join(pairs)
        more = "" if len(offenders) <= 5 else f"; {len(offenders) - 5} more"
        raise DatasetError(f"Image leakage across splits: {shown}{more}.")


# --- Internal helpers ---------------------------------------------------------


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    """Validate that `ratios` is a positive triple summing to 1.0."""
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(f"split ratios must be positive, got {ratios}.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}.")


def _assert_unique_image_ids(records: Sequence[ImageInfo]) -> None:
    """Reject duplicate image_ids so a source image cannot be split twice."""
    seen: set[str] = set()
    for info in records:
        if info.image_id in seen:
            raise DatasetError(
                f"duplicate image_id {info.image_id!r} in the input; cannot split unambiguously."
            )
        seen.add(info.image_id)
