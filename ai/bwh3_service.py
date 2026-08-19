"""
BWH3 (BitwiseHide 3) AI-guided steganography service (Phase 2.8.7).

Orchestrates the low-level BWH3 primitives from ai.bwh3_embedding and owns
image I/O (loading/saving PNGs). It integrates with the Phase 2.8.6
SuitabilityPredictor for both embedding and extraction.

This service is deliberately decoupled from:
- the encryption layer (the payload is opaque BYTES; confidentiality and
  tamper-evidence come from EncryptionService, never from this layer)
- API endpoints and FastAPI (pure Python, framework-agnostic)
- database repositories

Images and payloads are exchanged as raw bytes, mapping directly to the
future HTTP upload/download flow. PNG decoding/encoding is delegated to
Pillow — no manual PNG parsing.

Only PNG images are accepted; anything else fails closed with
BWH3EmbeddingError.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PIL import Image

from ai.bwh3_embedding import (
    BWH3EmbedConfig,
    BWH3EmbeddingError,
)
from ai.bwh3_embedding import (
    embed_bytes as bwh3_embed_bytes,
)
from ai.bwh3_embedding import (
    extract_bytes as bwh3_extract_bytes,
)
from ai.bwh3_embedding import (
    max_payload_bytes as bwh3_max_payload_bytes,
)

if TYPE_CHECKING:
    from ai.inference import SuitabilityPredictor


class BWH3SteganographyService:
    """Embed and extract opaque payload bytes in PNG images using AI-guided BWH3."""

    def __init__(self, predictor: SuitabilityPredictor) -> None:
        """
        Initialize the service with a SuitabilityPredictor.

        Args:
            predictor: The predictor bound to a ModelRegistry containing
                the AI suitability models to use for embedding/extraction.
        """
        self._predictor = predictor

    @property
    def predictor(self) -> SuitabilityPredictor:
        """The SuitabilityPredictor this service uses."""
        return self._predictor

    def embed(
        self,
        *,
        image: bytes,
        payload: bytes,
        model_name: str,
        model_version: str,
    ) -> bytes:
        """
        Embed `payload` into a PNG image, prioritizing AI-suitable pixels.

        Args:
            image: PNG file bytes to hide the payload in.
            payload: Opaque bytes to embed (e.g., EncryptedPayload.to_bytes()).
            model_name: Registry model name (the stable model identifier).
            model_version: Registry version identifier.

        Returns:
            PNG file bytes of a new image carrying the payload.

        Raises:
            BWH3EmbeddingError: If `image` is not a valid PNG or the
                payload does not fit within the image's capacity.
        """
        source = self._load_png(image)

        config = BWH3EmbedConfig(
            model_name=model_name,
            model_version=model_version,
            predictor=self._predictor,
        )

        stego = bwh3_embed_bytes(source, payload, config=config)
        return self._save_png(stego)

    def extract(
        self,
        image: bytes,
    ) -> bytes:
        """
        Extract the BWH3 payload from a PNG image.

        The embedding order is reproduced from the stego image alone: the header
        (read from the fixed leading positions) supplies the model identification
        and payload length. The suitability map is recomputed from the SAME model
        (loaded via the predictor using the header's model_name/version) on the
        image with its LSBs cleared — which is bit-identical to the embedder's
        analysis input, so the candidate positions match exactly.

        Args:
            image: PNG file bytes produced by `embed` (or a modified copy).

        Returns:
            The exact embedded payload bytes (opaque — caller validates them).

        Raises:
            BWH3EmbeddingError: If `image` is not a valid PNG or carries no
                valid BWH3 payload, or if the required model is not available
                in the predictor's registry.
        """
        source = self._load_png(image)
        return bwh3_extract_bytes(source, predictor=self._predictor)

    def max_payload_bytes(
        self,
        image: bytes,
        model_name: str,
        model_version: str,
    ) -> int:
        """
        Maximum payload bytes that fit in `image` with the given model.

        Args:
            image: PNG file bytes.
            model_name: Registry model name.
            model_version: Registry version identifier.

        Returns:
            The largest payload that embed will accept. May be negative for
            images too small to hold even the maximum header.
        """
        source = self._load_png(image)
        config = BWH3EmbedConfig(
            model_name=model_name,
            model_version=model_version,
            predictor=self._predictor,
        )
        return bwh3_max_payload_bytes(source, config)

    @staticmethod
    def _load_png(image: bytes) -> Image.Image:
        """Load PNG bytes into a PIL image, failing closed on anything else."""
        if not image:
            raise BWH3EmbeddingError(message="Image data is empty.")
        try:
            img = Image.open(io.BytesIO(image))
            img.load()
        except (OSError, ValueError, EOFError, Image.DecompressionBombError) as exc:
            raise BWH3EmbeddingError(message="Invalid image data.") from exc
        if img.format != "PNG":
            raise BWH3EmbeddingError(
                message=f"Only PNG images are supported, got {img.format or 'unknown'}."
            )
        return img

    @staticmethod
    def _save_png(image: Image.Image) -> bytes:
        """Encode a PIL image as PNG file bytes (lossless)."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
