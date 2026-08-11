"""
BSDS500-style dataset discovery, validation, and loading (Phase 2.8.1).

This is the training-time dataset boundary. It knows how to *find* a locally
downloaded BSDS500 tree, verify every image decodes as RGB, and hand back
stable per-image records (``ImageInfo``) plus a ``Dataset`` handle that the
splitter (``ai.split``) consumes. It deliberately does NOT download the dataset
and does NOT read payloads, labels, or anything from ``app.core``: the dataset
layer is framework-agnostic, deterministic, and separate from the production
backend.

Provenance
----------
The intended source is BSDS500 (Arbeláez, Maire, Fowlkes, Malik, "Contour
Detection and Hierarchical Image Segmentation", IEEE TPAMI 33(5), 2011),
500 natural images officially split 200 train / 100 val / 200 test. Download
is manual and out of scope for this module — see ``docs/dataset.md`` for the
exact source URL, license, expected directory layout, and integrity-verification
requirements. No download happens here, and no secrets/credentials are involved.

Accepted layouts (detected in this order)
-----------------------------------------
1. ``<root>/images/<train|val|test>/*.{jpg,...}``  (official split folders)
2. ``<root>/<train|val|test>/*.{jpg,...}``          (split folders at the root)
3. ``<root>/images/*.{jpg,...}`` plus ``iids_{train,val,test}.txt``
   (flat images with the official image-ID lists)
4. ``<root>/images/*.{jpg,...}``                    (flat, unsplit)

Every discovered image is decoded as RGB before the dataset is accepted; any
missing/corrupt/non-RGB file, any duplicate image ID, or any image not covered
by the id lists (layout 3) raises ``DatasetError`` so a broken download fails
loudly at prepare time instead of silently corrupting a training run.

Error contract mirrors ``app.core``: invalid *dataset state* raises
``DatasetError`` (the dataset is misconfigured); invalid *arguments* raise
``ValueError`` (the caller is misconfigured).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from ai.synthetic import SYNTHETIC_KINDS, noise_image, synthetic_image

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Repo-relative default root for downloaded datasets (git-ignored except
#: .gitkeep). Resolved from this module's location, so it works from any cwd.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"

#: Image extensions discovered as candidate source images (case-insensitive).
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

#: Canonical split folder names inside a BSDS500 layout.
SPLIT_DIRS = ("train", "val", "test")

#: Official BSDS500 source for documentation and provenance.
BSDS500_SOURCE_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500"
)
BSDS500_CITATION = "Arbeláez, Maire, Fowlkes & Malik (2011), IEEE TPAMI 33(5)."


class Split(StrEnum):
    """Canonical dataset split names, matching BSDS500's official folders."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


#: Lookup from folder name to Split, used when tagging discovered images.
_SPLIT_BY_NAME: dict[str, Split] = {split.value: split for split in Split}

#: Expected official BSDS500 image counts per split.
BSDS500_EXPECTED_COUNTS: dict[Split, int] = {
    Split.TRAIN: 200,
    Split.VAL: 100,
    Split.TEST: 200,
}


class DatasetError(Exception):
    """Raised when a dataset is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class ImageInfo:
    """A single source image in a dataset.

    Attributes:
        image_id: Stable identifier, unique within the dataset (filename stem,
            or the id given by an iids list).
        path: Absolute or caller-supplied path to the image file.
        source_split: Split the image belongs to in the official layout, or
            None when the dataset is unsplit.
    """

    image_id: str
    path: Path
    source_split: Split | None = None


@dataclass(frozen=True)
class Dataset:
    """A validated collection of source images ready to split.

    Attributes:
        name: Dataset name ("bsds500" or "synthetic").
        root: Root directory the dataset was discovered from.
        source_url, citation: Provenance for documentation and licensing.
        images: Validated image records.
        official_split_available: True when the intended BSDS500 train/val/test
            layout (folders or id lists) was found, so split resolution can
            adopt it instead of a fallback partition.
    """

    name: str
    root: Path
    source_url: str
    citation: str
    images: tuple[ImageInfo, ...]
    official_split_available: bool

    @property
    def size(self) -> int:
        """Total number of source images."""
        return len(self.images)

    def counts_by_split(self) -> dict[Split, int]:
        """Image counts per split, from each record's source_split."""
        counts: dict[Split, int] = dict.fromkeys(Split, 0)
        for info in self.images:
            if info.source_split is not None:
                counts[info.source_split] += 1
        return counts


def discover_bsds500(
    root: Path | None = None,
    *,
    name: str = "bsds500",
    source_url: str = BSDS500_SOURCE_URL,
    citation: str = BSDS500_CITATION,
) -> Dataset:
    """Discover and validate a local BSDS500-style image directory.

    Args:
        root: Directory to scan; defaults to the repo ``dataset/`` directory.
        name, source_url, citation: Provenance carried on the returned Dataset
            (override for synthetic datasets).

    Returns:
        A validated Dataset. When the official train/val/test layout is present,
        ``official_split_available`` is True and every ImageInfo carries its
        official split.

    Raises:
        DatasetError: If `root` does not exist, contains no images, has a
            partial split layout, or any image fails RGB validation; if any
            image_id is duplicated; or if id lists leave images uncovered.
    """
    root_path = Path(root) if root is not None else DEFAULT_DATASET_ROOT
    if not root_path.is_dir():
        msg = (
            f"Dataset root {root_path} does not exist. "
            "See docs/dataset.md for how to obtain BSDS500 locally."
        )
        raise DatasetError(msg)

    images_dir = root_path / "images" if (root_path / "images").is_dir() else root_path
    split_dirs = [d for d in SPLIT_DIRS if (images_dir / d).is_dir()]
    if split_dirs and len(split_dirs) != len(SPLIT_DIRS):
        msg = (
            f"Partial split directories under {images_dir}: found {split_dirs}, "
            f"expected all of {SPLIT_DIRS}."
        )
        raise DatasetError(msg)

    if len(split_dirs) == len(SPLIT_DIRS):
        infos = _discover_split_folders(images_dir)
        official = True
    else:
        iid_paths = _find_iid_lists(root_path, images_dir)
        if iid_paths is not None:
            infos = _discover_from_id_lists(images_dir, iid_paths)
            official = True
        else:
            infos = _discover_flat(images_dir)
            official = False

    if not infos:
        raise DatasetError(f"No images found under {images_dir}. See docs/dataset.md.")

    _validate_records(infos)
    return Dataset(
        name=name,
        root=root_path,
        source_url=source_url,
        citation=citation,
        images=tuple(infos),
        official_split_available=official,
    )


def verify_bsds500(
    dataset: Dataset,
    *,
    expected: dict[Split, int] = BSDS500_EXPECTED_COUNTS,
) -> None:
    """Verify the official BSDS500 split counts on `dataset`.

    Raises:
        DatasetError: If `dataset` has no official split or any split count
            differs from `expected`.
    """
    if not dataset.official_split_available:
        msg = "Cannot verify official BSDS500 counts on an unsplit dataset."
        raise DatasetError(msg)
    counts = dataset.counts_by_split()
    mismatched = [split.value for split in Split if counts[split] != expected[split]]
    if mismatched:
        msg = f"Split counts {counts} do not match the official {expected}."
        raise DatasetError(msg)


def load_image_rgb(path: Path) -> Image.Image:
    """Load `path` and normalize it to an RGB PIL image.

    Raises:
        DatasetError: If `path` is not a file, fails to decode, has non-positive
            dimensions, or cannot be normalized to RGB.
    """
    if not path.is_file():
        raise DatasetError(f"image file does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            if image.size[0] <= 0 or image.size[1] <= 0:
                raise ValueError(f"non-positive dimensions {image.size}")
            return image.convert("RGB")
    except Exception as exc:
        raise DatasetError(f"cannot load image {path} as RGB: {exc}") from exc


def validate_image_rgb(path: Path) -> None:
    """Validate that `path` decodes as a non-empty RGB image.

    Raises:
        DatasetError: If the image cannot be read or normalized to RGB.
    """
    load_image_rgb(path)


def write_synthetic_dataset(
    root: Path,
    *,
    per_split: int = 2,
    size: tuple[int, int] = (64, 64),
    seed: int = 0,
) -> Dataset:
    """Deterministically write a small synthetic dataset with the BSDS500 layout.

    Creates ``<root>/images/{train,val,test}/synthetic_{split}_{kind}_{i:04d}.png``
    cycling over every synthetic kind. Fully offline and reproducible in `seed`;
    intended for smoke tests and local runs without downloading BSDS500. The
    written files live under git-ignored paths (see .gitignore) and must not be
    committed.

    Args:
        root: Directory to write into (created if absent).
        per_split: Number of images per split.
        size: Pixel dimensions of every generated image.
        seed: Base seed; per-image seeds vary deterministically.

    Returns:
        The discovered synthetic Dataset (official split available).
    """
    images_dir = Path(root) / "images"
    for split_idx, split_name in enumerate(SPLIT_DIRS):
        split_dir = images_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_split):
            kind = SYNTHETIC_KINDS[(split_idx + i) % len(SYNTHETIC_KINDS)]
            filename = f"synthetic_{split_name}_{kind}_{i:04d}.png"
            image = _render_synthetic(kind, size=size, seed=seed + split_idx * 10000 + i)
            image.save(split_dir / filename, format="PNG")
    return discover_bsds500(
        root,
        name="synthetic",
        source_url="",
        citation="Generated locally by ai.synthetic; no external provenance.",
    )


# --- Internal helpers ---------------------------------------------------------


def _render_synthetic(kind: str, *, size: tuple[int, int], seed: int) -> Image.Image:
    """Render `kind` with its defaults; only noise depends on the seed."""
    if kind == "noise":
        return noise_image(size=size, seed=seed)
    return synthetic_image(kind, size=size)


def _iter_image_paths(directory: Path) -> Iterator[Path]:
    """Yield image files under `directory` (recursive, sorted, deterministic)."""
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            yield path


def _discover_split_folders(images_dir: Path) -> list[ImageInfo]:
    """Tag every image under <images_dir>/<train|val|test> with its split."""
    infos: list[ImageInfo] = []
    for split_name in SPLIT_DIRS:
        split = _SPLIT_BY_NAME[split_name]
        for path in _iter_image_paths(images_dir / split_name):
            infos.append(ImageInfo(image_id=path.stem, path=path, source_split=split))
    return infos


def _discover_flat(images_dir: Path) -> list[ImageInfo]:
    """Treat every image under `images_dir` as an unsplit source image."""
    return [ImageInfo(image_id=path.stem, path=path) for path in _iter_image_paths(images_dir)]


def _find_iid_lists(root: Path, images_dir: Path) -> list[Path] | None:
    """Locate the three iids files (official id lists), or None if absent."""
    for base in (images_dir, root):
        paths = [base / f"iids_{split}.txt" for split in SPLIT_DIRS]
        if all(path.is_file() for path in paths):
            return paths
    return None


def _read_image_id_list(path: Path) -> list[str]:
    """Read image IDs from an iids file (one id or ``id.jpg`` per line)."""
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(Path(line).stem)
    return ids


def _discover_from_id_lists(images_dir: Path, iid_paths: Sequence[Path]) -> list[ImageInfo]:
    """Build the official split from flat images plus the three iids lists.

    Every listed id must resolve to a file and every image file must be covered
    by exactly one list; otherwise the dataset is ambiguous and ``DatasetError``
    is raised (this is the flat-layout leakage guard).
    """
    by_stem = {path.stem: path for path in _iter_image_paths(images_dir)}
    infos: list[ImageInfo] = []
    assigned: set[str] = set()
    for iid_path, split_name in zip(iid_paths, SPLIT_DIRS, strict=True):
        split = _SPLIT_BY_NAME[split_name]
        for image_id in _read_image_id_list(iid_path):
            if image_id in assigned:
                msg = f"Image {image_id!r} is listed in multiple iids files; cannot split."
                raise DatasetError(msg)
            if image_id not in by_stem:
                msg = f"Image {image_id!r} from {iid_path.name} has no file under {images_dir}."
                raise DatasetError(msg)
            assigned.add(image_id)
            infos.append(ImageInfo(image_id=image_id, path=by_stem[image_id], source_split=split))
    unassigned = sorted(set(by_stem) - assigned)
    if unassigned:
        shown = ", ".join(unassigned[:5])
        more = "" if len(unassigned) <= 5 else f" and {len(unassigned) - 5} more"
        raise DatasetError(f"Images not covered by the iids lists: {shown}{more}.")
    return infos


def _validate_records(infos: Sequence[ImageInfo]) -> None:
    """Enforce unique image IDs and decode every image as RGB.

    Raises:
        DatasetError: On a duplicate image_id or any image that fails to load
            as RGB (all failures are reported together).
    """
    seen: set[str] = set()
    failures: list[tuple[Path, str]] = []
    for info in infos:
        if info.image_id in seen:
            raise DatasetError(f"duplicate image_id {info.image_id!r} at {info.path}.")
        seen.add(info.image_id)
        try:
            validate_image_rgb(info.path)
        except DatasetError as exc:
            failures.append((info.path, str(exc)))
    if failures:
        details = "; ".join(f"{path}: {reason}" for path, reason in failures[:5])
        more = "" if len(failures) <= 5 else f"; {len(failures) - 5} more"
        raise DatasetError(f"{len(failures)} image(s) failed RGB validation: {details}{more}.")
