"""
Steganography service — hides and retrieves opaque payload bytes inside PNG images.

Orchestrates the low-level LSB primitives from core.steganography and owns image
I/O (loading/saving PNGs). It is deliberately decoupled from:
- the encryption layer (the payload is opaque BYTES; confidentiality and
  tamper-evidence come from EncryptionService, never from this layer)
- API endpoints and FastAPI (pure Python, framework-agnostic)
- database repositories

Images and payloads are exchanged as raw bytes, which maps directly to the
future HTTP upload/download flow. PNG decoding/encoding is delegated to Pillow —
no manual PNG parsing.

Design decisions:
- Only PNG images are accepted; anything else fails closed with
  SteganographyError.
- Extracted bytes are returned UNVALIDATED — they are untrusted input. It is
  the caller's responsibility to feed them to EncryptedPayload.from_bytes()
  and EncryptionService.decrypt(), which apply their own fail-closed checks.
"""

from __future__ import annotations

import io

from PIL import Image

from app.core.exceptions import SteganographyError
from app.core.steganography import embed_bytes, extract_bytes


class SteganographyService:
    """Embed and extract opaque payload bytes in PNG images."""

    def embed(self, *, image: bytes, payload: bytes) -> bytes:
        """
        Embed `payload` into a PNG image.

        Args:
            image: PNG file bytes to hide the payload in.
            payload: Opaque bytes to embed (e.g., EncryptedPayload.to_bytes()).

        Returns:
            PNG file bytes of a new image carrying the payload.

        Raises:
            SteganographyError: If `image` is not a valid PNG or the payload
                does not fit within the image's capacity.
        """
        source = self._load_png(image)
        stego = embed_bytes(source, payload)
        return self._save_png(stego)

    def extract(self, image: bytes) -> bytes:
        """
        Extract the hidden payload from a PNG image.

        Args:
            image: PNG file bytes produced by `embed` (or a modified copy).

        Returns:
            The exact embedded payload bytes (opaque — caller validates them).

        Raises:
            SteganographyError: If `image` is not a valid PNG or carries no
                valid BitwiseHide payload.
        """
        source = self._load_png(image)
        return extract_bytes(source)

    @staticmethod
    def _load_png(image: bytes) -> Image.Image:
        """Load PNG bytes into a PIL image, failing closed on anything else."""
        if not image:
            raise SteganographyError(message="Image data is empty.")
        try:
            img = Image.open(io.BytesIO(image))
            img.load()
        except (OSError, ValueError, EOFError, Image.DecompressionBombError) as exc:
            raise SteganographyError(message="Invalid image data.") from exc
        if img.format != "PNG":
            raise SteganographyError(
                message=f"Only PNG images are supported, got {img.format or 'unknown'}."
            )
        return img

    @staticmethod
    def _save_png(image: Image.Image) -> bytes:
        """Encode a PIL image as PNG file bytes (lossless)."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
