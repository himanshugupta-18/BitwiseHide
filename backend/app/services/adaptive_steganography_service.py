"""
Adaptive steganography service — embeds and retrieves opaque payload bytes
inside PNG images using Phase 2.5's complexity analysis to pick hiding spots.

Orchestrates the low-level primitives from core.adaptive_embedding and owns
image I/O (loading/saving PNGs). It is deliberately decoupled from:
- the encryption layer (the payload is opaque BYTES; confidentiality and
  tamper-evidence come from EncryptionService, never from this layer)
- API endpoints and FastAPI (pure Python, framework-agnostic)
- database repositories

This mirrors the Phase 2.4 SteganographyService architecture:
- Images and payloads are exchanged as raw bytes, mapping directly to the
  future HTTP upload/download flow. PNG decoding/encoding is delegated to
  Pillow — no manual PNG parsing.
- Only PNG images are accepted; anything else fails closed.
- Extracted bytes are returned UNVALIDATED — they are untrusted input. It is
  the caller's responsibility to feed them to EncryptedPayload.from_bytes()
  and EncryptionService.decrypt(), which apply their own fail-closed checks.
"""

from __future__ import annotations

import io

from PIL import Image

from app.core.adaptive_embedding import embed_bytes, extract_bytes
from app.core.exceptions import AdaptiveEmbeddingError


class AdaptiveSteganographyService:
    """Embed and extract opaque payload bytes in PNG images, adaptively."""

    def embed(
        self,
        *,
        image: bytes,
        payload: bytes,
        edge_weight: float = 1.0,
        texture_weight: float = 1.0,
    ) -> bytes:
        """
        Embed `payload` into a PNG image, prioritizing complex regions.

        Args:
            image: PNG file bytes to hide the payload in.
            payload: Opaque bytes to embed (e.g., EncryptedPayload.to_bytes()).
            edge_weight: Sobel-edge weight for the Phase 2.5 analysis.
            texture_weight: Local-texture weight for the Phase 2.5 analysis.
                Both are recorded in the payload header so extraction
                reproduces the exact embedding order.

        Returns:
            PNG file bytes of a new image carrying the payload.

        Raises:
            AdaptiveEmbeddingError: If `image` is not a valid PNG or the
                payload does not fit within the image's capacity.
            ValueError: If the analysis weights are invalid.
        """
        source = self._load_png(image)
        stego = embed_bytes(
            source,
            payload,
            edge_weight=edge_weight,
            texture_weight=texture_weight,
        )
        return self._save_png(stego)

    def extract(self, image: bytes) -> bytes:
        """
        Extract the adaptive payload from a PNG image.

        Args:
            image: PNG file bytes produced by `embed` (or a modified copy).

        Returns:
            The exact embedded payload bytes (opaque — caller validates them).

        Raises:
            AdaptiveEmbeddingError: If `image` is not a valid PNG or carries no
                valid adaptive BitwiseHide payload.
        """
        source = self._load_png(image)
        return extract_bytes(source)

    @staticmethod
    def _load_png(image: bytes) -> Image.Image:
        """Load PNG bytes into a PIL image, failing closed on anything else."""
        if not image:
            raise AdaptiveEmbeddingError(message="Image data is empty.")
        try:
            img = Image.open(io.BytesIO(image))
            img.load()
        except (OSError, ValueError, EOFError, Image.DecompressionBombError) as exc:
            raise AdaptiveEmbeddingError(message="Invalid image data.") from exc
        if img.format != "PNG":
            raise AdaptiveEmbeddingError(
                message=f"Only PNG images are supported, got {img.format or 'unknown'}."
            )
        return img

    @staticmethod
    def _save_png(image: Image.Image) -> bytes:
        """Encode a PIL image as PNG file bytes (lossless)."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
