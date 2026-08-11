# Phase 2.8.1 — Training Dataset (BSDS500)

This document covers the training dataset for the BitwiseHide suitability
model: its provenance, expected on-disk layout, train/val/test separation, and
the rules for keeping datasets out of Git. The dataset is **training-time only**
and never touches the production backend.

## Provenance

The intended primary source is **BSDS500** — the Berkeley Segmentation Dataset
and Benchmark, 500 natural images.

- **Citation**: P. Arbeláez, M. Maire, C. Fowlkes, and J. Malik, *Contour
  Detection and Hierarchical Image Segmentation*, IEEE TPAMI 33(5), 2011.
- **Official page**:
  <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500>
- **License/terms**: freely available for research use; the paper must be
  cited. Commercial use requires permission from the authors.

### Provenance & licensing verification requirements

- Download only from the official page above (or the archive it points to) —
  do **not** use random mirrors or scraped copies.
- Verify the downloaded archive's integrity against the checksum published
  with the release (typically SHA-256). If no checksum is published, note the
  date/source of the download in your experiment log.
- Record which tarball/version you used and the download date; the training
  pipeline is deterministic given the exact input files, so provenance must be
  reproducible from that record.
- BSDS500 requires attribution — keep the citation visible in any artifacts
  derived from it (training runs, reports, model cards).

## Downloading (manual — the code never auto-downloads)

`ai.prepare_dataset` only *discovers and validates* a local tree; it never
downloads. Obtaining BSDS500 is a manual, terms-gated step:

1. Open the official page, download the BSDS500 archive.
2. Verify integrity (see above).
3. Extract so the image directories land under `dataset/` in this repo, e.g.:

   ```
   dataset/
   └── BSDS500/
       └── data/
           └── images/
               ├── train/   (200 *.jpg)
               ├── val/     (100 *.jpg)
               └── test/    (200 *.jpg)
   ```

## Expected directory structure

`ai.prepare_dataset.discover_bsds500()` accepts four layouts, detected in order:

| # | Layout | Official split? |
|---|--------|-----------------|
| 1 | `dataset/images/<train\|val\|test>/*.{jpg,...}` | yes (folders) |
| 2 | `dataset/<train\|val\|test>/*.{jpg,...}` | yes (folders at root) |
| 3 | `dataset/images/*.{jpg,...}` + `iids_{train,val,test}.txt` | yes (id lists) |
| 4 | `dataset/images/*.{jpg,...}` (flat, no metadata) | no (fallback) |

`iids_*.txt` files contain one image ID per line (either `100007` or
`100007.jpg`); lines starting with `#` and blank lines are ignored. In layout 3
every listed ID must resolve to a file and every image file must be covered by
exactly one list, otherwise discovery raises `DatasetError` — this is the
flat-layout leakage guard.

Every discovered image is decoded as **RGB** before the dataset is accepted.
Missing/corrupt/non-RGB files and duplicate image IDs raise `DatasetError`.

## Train / validation / test separation

- **Official split**: 200 train / 100 val / 200 test, used automatically when
  the split folders or id lists are present (`ai.split.resolve_split`).
- **Fallback**: a deterministic 40/20/40 image-level partition
  (`ai.split.split_dataset`, default seed 42) is used only for unsplit trees.
- Splits are always **image-level**: the same source image never appears in
  more than one split. Crops are produced in a later phase (the dataloader) and
  inherit this separation because they derive from a single split assignment.
- `ai.split.assert_no_leakage` is invoked on every partition and raises
  `DatasetError` if an image ID appears in two splits.

## Git — datasets must not be committed

- `dataset/` is git-ignored except the `.gitkeep` placeholder (see
  `.gitignore`), so downloaded BSDS500 content and locally generated synthetic
  smoke datasets are never tracked.
- Do **not** `git add -f` or otherwise force-commit dataset contents. BSDS500 is
  ~100 MB+; committing it bloats the repository and may violate its license.
- Synthetic smoke datasets are regenerated deterministically on demand; they
  carry no provenance value worth committing.

## Verification

- `ai.prepare_dataset.discover_bsds500()` decodes every image as RGB and raises
  `DatasetError` on any missing/corrupt/non-RGB file or duplicate image ID.
- `ai.prepare_dataset.verify_bsds500(dataset)` additionally checks the official
  split counts against 200/100/200.

## Synthetic smoke dataset (offline)

To exercise the pipeline without downloading BSDS500:

```python
from ai.prepare_dataset import write_synthetic_dataset

ds = write_synthetic_dataset("dataset/synthetic", per_split=3, size=(64, 64), seed=0)
# -> dataset/synthetic/images/{train,val,test}/synthetic_{split}_{kind}_0000.png
```

`write_synthetic_dataset` deterministically writes smooth, checkerboard, noise,
gradient, and iso-luminant color-texture images (see `ai.synthetic`) and returns
the discovered Dataset with the official split available. All generated files
are git-ignored.
