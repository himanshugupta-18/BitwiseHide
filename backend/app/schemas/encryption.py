"""
Encrypted-payload schema — the versioned wire format for protected credentials.

Architecture decisions:
- The payload is a self-describing, versioned JSON object carrying every piece
  of metadata needed to decrypt without any external state:
      {
        "version": 1,
        "kdf": {"name": "scrypt", "n": ..., "r": ..., "p": ...},
        "salt": "<base64>",
        "nonce": "<base64>",
        "ciphertext": "<base64>",
        "tag": "<base64>"
      }
- `salt`/`nonce`/`ciphertext`/`tag` are stored as base64 strings — the JSON
  wire form is exactly what Phase 2.4 will embed into an image.
- KDF parameters live INSIDE the payload so a future scrypt-cost increase never
  locks out previously written payloads (each payload is decryptable on its own).
- `extra="forbid"` rejects unknown fields at parse time — the format is strict
  and fails closed on anything it does not understand.
- The numeric KDF ranges are intentionally NOT validated here; bounds checking
  lives in one place (core.crypto.validate_scrypt_params) and is applied by the
  EncryptionService at decrypt time.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KDFParams(BaseModel):
    """Key-derivation-function parameters stored with the payload."""

    name: Literal["scrypt"] = "scrypt"
    n: int  # CPU/memory cost (power of two)
    r: int  # block size
    p: int  # parallelization


class EncryptedPayload(BaseModel):
    """Self-describing encrypted payload (versioned, tamper-evident)."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(..., ge=1, description="Payload format version")
    kdf: KDFParams
    salt: str  # base64 — 16 bytes of fresh per-operation randomness
    nonce: str  # base64 — 12-byte GCM nonce, unique per encryption
    ciphertext: str  # base64 — AES-256-GCM ciphertext (tag removed)
    tag: str  # base64 — 16-byte GCM authentication tag

    def to_bytes(self) -> bytes:
        """
        Serialize the payload to bytes for storage/embedding (Phase 2.4).

        Returns:
            UTF-8 JSON representation of the payload.
        """
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> EncryptedPayload:
        """Deserialize a payload previously produced by `to_bytes`."""
        return cls.model_validate_json(data)
