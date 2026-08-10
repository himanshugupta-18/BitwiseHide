"""
Evaluation service — deterministic comparison of steganography methods.

Orchestrates Phase 2.7's objective comparison between Phase 2.4's basic LSB
steganography and Phase 2.6's adaptive LSB steganography. It REUSES the two
production services — SteganographyService and AdaptiveSteganographyService —
verbatim; it does not re-implement any embedding logic. Metrics come from
core.evaluation (pure, deterministic). This layer is deliberately decoupled
from FastAPI, the database, and the frontend: it consumes and produces raw
bytes and plain dataclasses.

Architecture decisions:
- SINGLE comparison entry point: ``compare_methods`` runs both methods on the
  same cover + payload and returns a ``ComparisonResult``. The payload is
  treated as opaque bytes (exactly as the Phase 2.4/2.6 layers do), so any
  payload — including an encrypted EncryptedPayload from Phase 2.3 — can be
  evaluated unchanged.
- FAILURE IS DATA, NOT EXCEPTIONS: a payload that exceeds a method's capacity
  is recorded (fits=False, quality=None) rather than raised, so a comparison
  still reports BOTH methods side by side. Only genuinely invalid input
  (a cover that is not a decodable PNG) raises EvaluationError.
- EXACT-RECOVERY CHECK: after embedding, the payload is extracted back and
  compared byte-for-byte. extracted_correctly is the ground-truth correctness
  flag — never an assumption.
- DETERMINISM: embedding, extraction, and all metrics are deterministic, so
  repeated calls on the same inputs yield identical results (the reported
  wall-clock runtimes are the only fields that vary, and they are isolated).
- No claims are made about which method is "better" — the numbers are
  returned and left to the consumer.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

from app.core import adaptive_embedding
from app.core import steganography as phase24
from app.core.evaluation import ImageQualityMetrics, image_quality
from app.core.exceptions import (
    AdaptiveEmbeddingError,
    EvaluationError,
    SteganographyError,
)
from app.services.adaptive_steganography_service import AdaptiveSteganographyService
from app.services.steganography_service import SteganographyService


@dataclass(frozen=True)
class EmbeddingEvaluation:
    """
    Per-method evaluation of one cover/payload pair.

    Attributes:
        method: Machine-readable method name ("basic_lsb" | "adaptive_lsb").
        capacity_bytes: Maximum payload bytes this method could embed.
        payload_size: Number of payload bytes supplied.
        fits: True iff the payload fit and embedding completed.
        extracted_correctly: True iff extraction returned the exact payload.
        embed_seconds / extract_seconds: Wall-clock runtimes in seconds.
        quality: cover-vs-stego MSE/PSNR/SSIM; None when `fits` is False.
        stego_png: PNG bytes of the produced stego image; None if embedding
            failed, so callers can inspect or re-verify the artifact.
    """

    method: str
    capacity_bytes: int
    payload_size: int
    fits: bool
    extracted_correctly: bool
    embed_seconds: float
    extract_seconds: float
    quality: ImageQualityMetrics | None
    stego_png: bytes | None


@dataclass(frozen=True)
class ComparisonResult:
    """
    Evaluation of basic LSB and adaptive LSB over one cover/payload pair.

    Attributes:
        cover_size: (width, height) of the cover image.
        payload_size: Number of payload bytes supplied.
        basic: Evaluation of Phase 2.4 basic LSB.
        adaptive: Evaluation of Phase 2.6 adaptive LSB.
    """

    cover_size: tuple[int, int]
    payload_size: int
    basic: EmbeddingEvaluation
    adaptive: EmbeddingEvaluation


class EvaluationService:
    """
    Deterministic, framework-agnostic evaluation of embedding methods.
    """

    def __init__(
        self,
        *,
        basic_service: SteganographyService | None = None,
        adaptive_service: AdaptiveSteganographyService | None = None,
    ) -> None:
        self._basic = basic_service or SteganographyService()
        self._adaptive = adaptive_service or AdaptiveSteganographyService()

    def compare_methods(
        self,
        *,
        cover: bytes,
        payload: bytes,
        edge_weight: float = 1.0,
        texture_weight: float = 1.0,
    ) -> ComparisonResult:
        """
        Run Phase 2.4 basic LSB and Phase 2.6 adaptive LSB on the same cover.

        Args:
            cover: PNG file bytes to hide the payload in.
            payload: Opaque payload bytes (e.g., EncryptedPayload.to_bytes()).
            edge_weight: Sobel-edge weight for the Phase 2.5 analysis
                (adaptive method only).
            texture_weight: Local-variance weight for the Phase 2.5 analysis
                (adaptive method only).

        Returns:
            A ComparisonResult with one EmbeddingEvaluation per method.

        Raises:
            EvaluationError: If `cover` is not a decodable PNG.
        """
        image = self._load_png(cover)
        return ComparisonResult(
            cover_size=image.size,
            payload_size=len(payload),
            basic=self._evaluate_basic(cover_bytes=cover, image=image, payload=payload),
            adaptive=self._evaluate_adaptive(
                cover_bytes=cover,
                image=image,
                payload=payload,
                edge_weight=edge_weight,
                texture_weight=texture_weight,
            ),
        )

    def _evaluate_basic(
        self,
        *,
        cover_bytes: bytes,
        image: Image.Image,
        payload: bytes,
    ) -> EmbeddingEvaluation:
        capacity = phase24.max_payload_bytes(image)
        start = time.perf_counter()
        try:
            stego_png = self._basic.embed(image=cover_bytes, payload=payload)
            embed_seconds = time.perf_counter() - start
        except SteganographyError:
            return EmbeddingEvaluation(
                method="basic_lsb",
                capacity_bytes=capacity,
                payload_size=len(payload),
                fits=False,
                extracted_correctly=False,
                embed_seconds=time.perf_counter() - start,
                extract_seconds=0.0,
                quality=None,
                stego_png=None,
            )
        extract_seconds, extracted_correctly, stego_image = self._extract(
            extract=self._basic.extract, stego_png=stego_png, payload=payload
        )
        quality = image_quality(image, stego_image)
        return EmbeddingEvaluation(
            method="basic_lsb",
            capacity_bytes=capacity,
            payload_size=len(payload),
            fits=True,
            extracted_correctly=extracted_correctly,
            embed_seconds=embed_seconds,
            extract_seconds=extract_seconds,
            quality=quality,
            stego_png=stego_png,
        )

    def _evaluate_adaptive(
        self,
        *,
        cover_bytes: bytes,
        image: Image.Image,
        payload: bytes,
        edge_weight: float,
        texture_weight: float,
    ) -> EmbeddingEvaluation:
        capacity = adaptive_embedding.max_payload_bytes(image)
        start = time.perf_counter()
        try:
            stego_png = self._adaptive.embed(
                image=cover_bytes,
                payload=payload,
                edge_weight=edge_weight,
                texture_weight=texture_weight,
            )
            embed_seconds = time.perf_counter() - start
        except AdaptiveEmbeddingError:
            return EmbeddingEvaluation(
                method="adaptive_lsb",
                capacity_bytes=capacity,
                payload_size=len(payload),
                fits=False,
                extracted_correctly=False,
                embed_seconds=time.perf_counter() - start,
                extract_seconds=0.0,
                quality=None,
                stego_png=None,
            )
        extract_seconds, extracted_correctly, stego_image = self._extract(
            extract=self._adaptive.extract, stego_png=stego_png, payload=payload
        )
        quality = image_quality(image, stego_image)
        return EmbeddingEvaluation(
            method="adaptive_lsb",
            capacity_bytes=capacity,
            payload_size=len(payload),
            fits=True,
            extracted_correctly=extracted_correctly,
            embed_seconds=embed_seconds,
            extract_seconds=extract_seconds,
            quality=quality,
            stego_png=stego_png,
        )

    @staticmethod
    def _load_png(image: bytes) -> Image.Image:
        """Decode PNG bytes into a PIL image, failing closed."""
        if not image:
            msg = "Image data is empty."
            raise EvaluationError(message=msg)
        try:
            img = Image.open(io.BytesIO(image))
            img.load()
        except (OSError, ValueError, EOFError, Image.DecompressionBombError) as exc:
            msg = "Invalid image data."
            raise EvaluationError(message=msg) from exc
        if img.format != "PNG":
            msg = f"Only PNG images are supported, got {img.format or 'unknown'}."
            raise EvaluationError(message=msg)
        return img

    def _extract(
        self,
        *,
        extract: Callable[[bytes], bytes],
        stego_png: bytes,
        payload: bytes,
    ) -> tuple[float, bool, Image.Image]:
        """
        Extract `payload` back from `stego_png` and compare it byte-for-byte.

        Returns (extract_seconds, extracted_correctly, stego_image). Any
        extraction failure records extracted_correctly=False instead of
        propagating — the stego image may still be valid enough to measure.
        """
        start = time.perf_counter()
        try:
            extracted = extract(stego_png)
        except (SteganographyError, AdaptiveEmbeddingError):
            extract_seconds = time.perf_counter() - start
            return extract_seconds, False, self._load_png(stego_png)
        extract_seconds = time.perf_counter() - start
        return extract_seconds, extracted == payload, self._load_png(stego_png)
