# Phase 2.7 — Evaluation & Baseline Comparison

The evaluation layer objectively compares the two completed steganography
methods on identical cover images and payloads:

1. **Phase 2.4 — Basic LSB** (row-major least-significant-bit embedding, `BWH1`)
2. **Phase 2.6 — Adaptive LSB** (complexity-ranked embedding, `BWH2`)

Everything runs on generated, in-memory images — no datasets, no external
services. The layer is framework-agnostic: it lives outside FastAPI, the
database, and the frontend, and consumes/produces raw PNG bytes and plain
dataclasses.

## What each metric means and why it matters for steganography

### MSE — Mean Squared Error

The average squared difference between the **original RGB pixel values** and
the **resulting stego RGB pixel values**:

```
MSE = (1 / (H·W·3)) · Σ (cover_pixel − stego_pixel)²
```

- `0` means the images are pixel-identical.
- Every hidden bit flips one pixel-byte by ±1, so MSE rises by exactly one
  unit per flipped byte. It is the raw fidelity floor.
- **Why it matters:** it is the direct, unambiguous measure of how much the
  cover image was altered. Everything else in image quality builds on it.

### PSNR — Peak Signal-to-Noise Ratio

A logarithmic rescaling of MSE:

```
PSNR = 10 · log10(MAX² / MSE),   MAX = 255 for 8-bit images
```

- Higher is better; `∞` when MSE is `0` (identical images — the exact,
  mathematically correct answer).
- `MAX` is the peak pixel value of the signal (255 for uint8 RGB), so PSNR
  is always derived from MSE with the correct denominator.
- **Why it matters:** PSNR is the de-facto fidelity metric in image
  processing. The log scale makes small MSE differences readable in dB
  (each ~6 dB ≈ a factor of 4 in MSE), which is why steganography papers
  report imperceptibility in PSNR.

### SSIM — Structural Similarity (Wang et al., 2004)

Compares **luminance, contrast, and structure** over local Gaussian windows:

```
SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
```

- Range `[-1, 1]`; `1` means perceptually identical.
- Unlike MSE/PSNR it is **perception-motivated**: a ±1 change in a flat
  region is far more visible than the same change inside textured noise,
  and SSIM reflects that via the local contrast (`σ`) normalization.
- **Why it matters:** this is the metric adaptive steganography should
  actually win on. Phase 2.5/2.6 exist to place bits where they are least
  visible (complex, textured regions first, smooth regions last); SSIM is
  the closest cheap proxy for "least visible." A global PSNR improvement is
  *not* expected from adaptive embedding (the bit count is identical), but
  a **localized/structural** improvement is — SSIM is how that shows up.

### Payload capacity & payload size

- `capacity_bytes`: the maximum payload this method can hide in the cover,
  including its own header. Phase 2.4's 12-byte `BWH1` header and Phase
  2.6's 14-byte `BWH2` header both reduce capacity; adaptive therefore
  carries two fewer payload bytes at full capacity.
- `payload_size`: the number of bytes actually supplied for this run.
- **Why it matters:** capacity is the real cost of a stego format. Any
  comparison that only looks at quality ignores that a heavier header or a
  different bit layout can shrink what fits.

### Extraction correctness

`extracted_correctly` is the ground truth: after embedding, the payload is
extracted back and compared byte-for-byte. It is never assumed.
**Why it matters:** a method that looks great in PSNR but cannot round-trip
the payload is not a method at all.

### Runtime

Wall-clock seconds for embed and extract (`time.perf_counter`). Runtime is
reported but inherently varies between machines; it is intentionally
isolated so it never affects the deterministic metric fields.

## How the comparison runs

`EvaluationService.compare_methods(cover=..., payload=...)` runs Phase 2.4
and Phase 2.6 on the **same cover + same payload**, then returns a
`ComparisonResult` with one `EmbeddingEvaluation` per method, each carrying
capacity, payload size, fits, extraction correctness, runtimes, quality
(MSE/PSNR/SSIM), and the produced stego PNG bytes.

The comparison **records** out-of-capacity failures (`fits=False`) instead
of raising, so both methods are always reported side by side. Only invalid
cover input (not a decodable PNG) raises `EvaluationError`.

The framework makes **no claim** about which method is better — it returns
the measurements. Conclusions are drawn from the numbers, not asserted.

## Implementation notes

- Metrics are implemented in `core/evaluation.py` with **pure numpy**
  (numpy is the only new dependency, added for Phase 2.7; SSIM's Gaussian
  filtering is hand-implemented so no scipy/scikit-image is needed).
- All metrics are deterministic; embedding is deterministic; extraction is
  deterministic. Repeated runs on identical inputs produce identical results
  except for the isolated runtime fields.
- Both methods must round-trip the exact payload — verified by the byte
  comparison, not by assumption.
