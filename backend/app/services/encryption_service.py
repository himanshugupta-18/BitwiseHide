"""
Encryption service — protects sensitive credential data at rest.

Orchestrates the low-level primitives from core.crypto and owns the payload
format from schemas.encryption. It is deliberately decoupled from:
- the authentication service (no cross-imports)
- API endpoints and FastAPI (pure Python, framework-agnostic)
- database repositories
- the future steganography layer (it consumes/produces EncryptedPayload bytes)

Design decisions:
- A FRESH random salt and GCM nonce are generated for every encryption, so the
  same password+plaintext NEVER produces the same payload (no nonce reuse).
- The entire payload header (version, KDF name/params, salt, nonce) is bound as
  GCM associated data. Tampering with ANY field therefore breaks authentication
  and decryption fails closed with EncryptionError.
- At decrypt time KDF parameters are bounds-checked BEFORE any key derivation,
  so a tampered payload cannot force an unbounded scrypt allocation.
- All failure modes (wrong password, tampering, malformed payload, unsupported
  version, invalid base64) raise EncryptionError — corrupted plaintext is never
  returned.
- scrypt cost parameters are explicit module defaults, overridable through the
  constructor. They are stored inside each payload, so a future cost bump never
  locks out older payloads.
"""

from __future__ import annotations

import base64
import binascii
import struct

from cryptography.exceptions import InvalidTag

from app.core.crypto import (
    GCM_NONCE_SIZE,
    SCRYPT_SALT_SIZE,
    aes256gcm_decrypt,
    aes256gcm_encrypt,
    derive_scrypt_key,
    generate_random_bytes,
    validate_scrypt_params,
)
from app.core.exceptions import EncryptionError
from app.schemas.encryption import EncryptedPayload, KDFParams

# The only payload format version this service understands.
_PAYLOAD_VERSION = 1

# Recommended scrypt cost parameters. Overridable via the constructor; stored
# per-payload so a future change here is backward compatible.
_DEFAULT_SCRYPT_N = 2**14
_DEFAULT_SCRYPT_R = 8
_DEFAULT_SCRYPT_P = 1


def _encode_header(
    *,
    version: int,
    kdf_name: str,
    n: int,
    r: int,
    p: int,
    salt: bytes,
    nonce: bytes,
) -> bytes:
    """
    Deterministic, unambiguous encoding of the payload header.

    The same bytes are produced during encryption and decryption, so the header
    can be authenticated as GCM associated data. Length prefixes and fixed-width
    integers make the encoding collision-free.
    """
    parts = [
        struct.pack(">B", version),
        len(kdf_name).to_bytes(4, "big"),
        kdf_name.encode("ascii"),
        struct.pack(">QQQ", n, r, p),
        len(salt).to_bytes(4, "big"),
        salt,
        len(nonce).to_bytes(4, "big"),
        nonce,
    ]
    return b"".join(parts)


def _b64decode(value: str) -> bytes:
    """Strict base64 decode (rejects non-base64 characters)."""
    return base64.b64decode(value, validate=True)


class EncryptionService:
    """
    Encrypt and decrypt credential data with AES-256-GCM + scrypt.

    Usage:
        service = EncryptionService()
        payload = service.encrypt(plaintext="my-secret", password="master-pass")
        stored = payload.to_bytes()          # hand to Phase 2.4 for embedding
        restored = service.decrypt(stored, "master-pass")  # == "my-secret"
    """

    def __init__(
        self,
        *,
        scrypt_n: int = _DEFAULT_SCRYPT_N,
        scrypt_r: int = _DEFAULT_SCRYPT_R,
        scrypt_p: int = _DEFAULT_SCRYPT_P,
    ) -> None:
        """
        Args:
            scrypt_n, scrypt_r, scrypt_p: scrypt cost parameters. Values are
                validated against the bounds in core.crypto; a ValueError is
                raised for out-of-range configuration.

        Raises:
            ValueError: If the scrypt parameters are outside accepted bounds.
        """
        validate_scrypt_params(n=scrypt_n, r=scrypt_r, p=scrypt_p)
        self._n = scrypt_n
        self._r = scrypt_r
        self._p = scrypt_p

    def encrypt(self, *, plaintext: str, password: str) -> EncryptedPayload:
        """
        Encrypt a plaintext string into a self-describing EncryptedPayload.

        A fresh salt and nonce are generated for this operation only. The caller
        is responsible for supplying a strong master password; the service does
        not enforce password policy (that is the callers' concern).

        Args:
            plaintext: The secret to protect (UTF-8).
            password: The master password used to derive the AES-256 key.

        Returns:
            A versioned payload ready to be persisted or embedded (Phase 2.4).

        Raises:
            EncryptionError: If the operation cannot complete safely.
        """
        salt = generate_random_bytes(SCRYPT_SALT_SIZE)
        nonce = generate_random_bytes(GCM_NONCE_SIZE)
        aad = _encode_header(
            version=_PAYLOAD_VERSION,
            kdf_name="scrypt",
            n=self._n,
            r=self._r,
            p=self._p,
            salt=salt,
            nonce=nonce,
        )
        key = derive_scrypt_key(password, salt, n=self._n, r=self._r, p=self._p)
        ciphertext, tag = aes256gcm_encrypt(key, nonce, plaintext.encode("utf-8"), aad)

        return EncryptedPayload(
            version=_PAYLOAD_VERSION,
            kdf=KDFParams(name="scrypt", n=self._n, r=self._r, p=self._p),
            salt=base64.b64encode(salt).decode("ascii"),
            nonce=base64.b64encode(nonce).decode("ascii"),
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
            tag=base64.b64encode(tag).decode("ascii"),
        )

    def decrypt(
        self,
        payload: EncryptedPayload | bytes | str | dict[str, object],
        password: str,
    ) -> str:
        """
        Decrypt a payload and return the original plaintext.

        Accepts the payload as an EncryptedPayload instance, raw serialized
        bytes/JSON string (e.g., the bytes produced by `to_bytes`), or a dict.
        Decryption verifies the GCM authentication tag and the header AAD, so
        any modification or wrong password raises EncryptionError.

        Args:
            payload: The encrypted payload (model, bytes, str, or dict).
            password: The master password used during encryption.

        Returns:
            The original plaintext string.

        Raises:
            EncryptionError: If the payload is malformed, tampered with, from an
                unsupported version, or the password is wrong. Corrupted
                plaintext is never returned.
        """
        parsed = self._parse_payload(payload)

        if parsed.version != _PAYLOAD_VERSION:
            msg = f"Unsupported payload version: {parsed.version}."
            raise EncryptionError(message=msg)

        try:
            salt = _b64decode(parsed.salt)
            nonce = _b64decode(parsed.nonce)
            ciphertext = _b64decode(parsed.ciphertext)
            tag = _b64decode(parsed.tag)
        except (binascii.Error, ValueError) as exc:
            raise EncryptionError(
                message="Encrypted payload contains invalid base64 data."
            ) from exc

        # Validate KDF parameters BEFORE deriving any key — a tampered payload
        # must not be able to force an unbounded scrypt memory/CPU allocation.
        try:
            validate_scrypt_params(n=parsed.kdf.n, r=parsed.kdf.r, p=parsed.kdf.p)
        except ValueError as exc:
            raise EncryptionError(message="Encrypted payload has invalid KDF parameters.") from exc

        aad = _encode_header(
            version=parsed.version,
            kdf_name=parsed.kdf.name,
            n=parsed.kdf.n,
            r=parsed.kdf.r,
            p=parsed.kdf.p,
            salt=salt,
            nonce=nonce,
        )

        try:
            key = derive_scrypt_key(
                password,
                salt,
                n=parsed.kdf.n,
                r=parsed.kdf.r,
                p=parsed.kdf.p,
            )
            plaintext_bytes = aes256gcm_decrypt(key, nonce, ciphertext, tag, aad)
        except InvalidTag as exc:
            raise EncryptionError(
                message="Decryption failed: wrong password or tampered data."
            ) from exc
        except (ValueError, MemoryError) as exc:
            raise EncryptionError(message="Decryption failed.") from exc

        try:
            return plaintext_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Authenticated plaintext should always be valid UTF-8, but never
            # let a decode failure surface as a raw exception.
            raise EncryptionError(message="Decrypted plaintext is not valid UTF-8.") from exc

    @staticmethod
    def _parse_payload(
        payload: EncryptedPayload | bytes | str | dict[str, object],
    ) -> EncryptedPayload:
        """Coerce any accepted payload representation into EncryptedPayload."""
        if isinstance(payload, EncryptedPayload):
            return payload
        try:
            if isinstance(payload, bytes):
                return EncryptedPayload.from_bytes(payload)
            if isinstance(payload, str):
                return EncryptedPayload.model_validate_json(payload)
            if isinstance(payload, dict):
                return EncryptedPayload.model_validate(payload)
        except (ValueError, TypeError) as exc:
            raise EncryptionError(message="Encrypted payload is malformed.") from exc
        raise EncryptionError(message="Unsupported payload type.")
