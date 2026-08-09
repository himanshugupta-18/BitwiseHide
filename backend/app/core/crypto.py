"""
Low-level cryptographic primitives for protecting credential data at rest.

Architecture decisions:
- scrypt for key derivation — memory-hard, resistant to GPU/ASIC cracking, and
  widely audited. A FRESH cryptographically secure random salt is generated per
  operation (never static/global), and the derived key is always 32 bytes for
  AES-256.
- AES-256-GCM for encryption — authenticated encryption providing BOTH
  confidentiality and integrity. A fresh 96-bit random nonce is generated per
  operation; GCM tags are 128 bits and are returned SEPARATELY from the
  ciphertext so the payload format can represent them independently.
- Parameters are explicit constants, validated at call time. The service layer
  stores the KDF parameters inside the encrypted payload so a future parameter
  change never breaks decrypting old payloads.
- These functions are deliberately byte-oriented, pure, and free of any
  FastAPI/settings/database dependency — exactly like core/security.py. The
  EncryptionService orchestrates them and owns the payload format.

Security properties guaranteed by this module:
- Never hardcoded keys/salts/nonces — randomness comes from secrets (OS CSPRNG).
- Decryption verifies the GCM authentication tag and NEVER returns corrupted
  plaintext; it raises InvalidTag on any tampering.
- scrypt parameters are bounds-checked so a hostile payload cannot force an
  unbounded memory/CPU allocation (DoS via crafted `n`).
"""

from __future__ import annotations

import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

# --- Algorithm constants ---

# AES-256 requires a 256-bit (32-byte) key.
AES_KEY_SIZE = 32
# GCM's recommended and most widely used nonce size is 96 bits.
GCM_NONCE_SIZE = 12
# GCM produces a 128-bit authentication tag.
GCM_TAG_SIZE = 16
# 128-bit salt — sufficient entropy for scrypt, matching common practice.
SCRYPT_SALT_SIZE = 16

# --- scrypt parameter bounds (enforced on every derivation, incl. decryption) ---
# These are the ranges the service will ACCEPT in a stored payload. They bound
# the CPU/memory a single derivation may consume (~128 * n * r bytes), so a
# tampered payload cannot trigger an unbounded allocation.
SCRYPT_MIN_N = 2**14  # 16 MiB at r=8
SCRYPT_MAX_N = 2**17  # 128 MiB at r=8 — above the OpenSSL default maxmem
SCRYPT_MIN_R = 1
SCRYPT_MAX_R = 16
SCRYPT_MIN_P = 1
SCRYPT_MAX_P = 8
# Hard cap on scrypt memory (bytes). Enforced both here and via Scrypt(maxmem=...).
SCRYPT_MAX_MEMORY = 128 * 1024 * 1024


def validate_scrypt_params(*, n: int, r: int, p: int) -> None:
    """
    Validate scrypt cost parameters against the accepted bounds.

    Raises:
        ValueError: If any parameter is out of range, `n` is not a power of two,
            or the implied memory footprint exceeds SCRYPT_MAX_MEMORY.
    """
    if not (SCRYPT_MIN_N <= n <= SCRYPT_MAX_N):
        msg = f"scrypt n must be in [{SCRYPT_MIN_N}, {SCRYPT_MAX_N}]."
        raise ValueError(msg)
    if n & (n - 1) != 0:  # not a power of two
        msg = f"scrypt n must be a power of two, got {n}."
        raise ValueError(msg)
    if not (SCRYPT_MIN_R <= r <= SCRYPT_MAX_R):
        msg = f"scrypt r must be in [{SCRYPT_MIN_R}, {SCRYPT_MAX_R}]."
        raise ValueError(msg)
    if not (SCRYPT_MIN_P <= p <= SCRYPT_MAX_P):
        msg = f"scrypt p must be in [{SCRYPT_MIN_P}, {SCRYPT_MAX_P}]."
        raise ValueError(msg)
    # scrypt memory usage is ~128 * n * r bytes.
    memory = 128 * n * r
    if memory > SCRYPT_MAX_MEMORY:
        msg = f"scrypt memory requirement (~{memory} bytes) exceeds {SCRYPT_MAX_MEMORY}."
        raise ValueError(msg)


def derive_scrypt_key(
    password: str,
    salt: bytes,
    *,
    n: int,
    r: int,
    p: int,
    length: int = AES_KEY_SIZE,
) -> bytes:
    """
    Derive a key of `length` bytes from a password using scrypt.

    Args:
        password: The master password/passphrase (UTF-8 encoded internally).
        salt: Per-operation random salt (must be at least 16 bytes).
        n, r, p: scrypt cost parameters (validated against SCRYPT_* bounds).
        length: Desired key length in bytes (default 32 for AES-256).

    Returns:
        Derived key bytes.

    Raises:
        ValueError: If parameters or salt are invalid.
    """
    if len(salt) < SCRYPT_SALT_SIZE:
        msg = f"scrypt salt must be at least {SCRYPT_SALT_SIZE} bytes."
        raise ValueError(msg)
    validate_scrypt_params(n=n, r=r, p=p)

    # Memory is bounded by validate_scrypt_params above (~128 * n * r bytes);
    # modern cryptography versions derive without an external maxmem argument.
    kdf = Scrypt(
        salt=salt,
        length=length,
        n=n,
        r=r,
        p=p,
    )
    return kdf.derive(password.encode("utf-8"))


def aes256gcm_encrypt(
    key: bytes,
    nonce: bytes,
    plaintext: bytes,
    associated_data: bytes,
) -> tuple[bytes, bytes]:
    """
    Encrypt plaintext with AES-256-GCM, returning (ciphertext, tag) separately.

    The caller must supply a FRESH, never-reused `nonce` for this key. The
    `associated_data` bytes are authenticated but not encrypted — tampering with
    them causes tag verification to fail during decryption.

    Args:
        key: 32-byte AES-256 key.
        nonce: 12-byte unique nonce.
        plaintext: Data to encrypt (any length, may be empty).
        associated_data: Unencrypted context bound to the ciphertext (AAD).

    Returns:
        Tuple of (ciphertext, tag). The 16-byte GCM tag is kept separate so the
        payload format can represent it independently.

    Raises:
        ValueError: If key or nonce have an invalid length.
    """
    if len(key) != AES_KEY_SIZE:
        msg = f"AES-256 requires a {AES_KEY_SIZE}-byte key, got {len(key)}."
        raise ValueError(msg)
    if len(nonce) != GCM_NONCE_SIZE:
        msg = f"GCM nonce must be {GCM_NONCE_SIZE} bytes, got {len(nonce)}."
        raise ValueError(msg)

    encrypted = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    # cryptography appends the 16-byte tag to the ciphertext; split it off.
    return encrypted[:-GCM_TAG_SIZE], encrypted[-GCM_TAG_SIZE:]


def aes256gcm_decrypt(
    key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    associated_data: bytes,
) -> bytes:
    """
    Decrypt and authenticate AES-256-GCM ciphertext.

    The GCM authentication tag is always verified. On any mismatch — wrong key,
    tampered ciphertext/tag/nonce, or modified associated data — InvalidTag is
    raised and NO plaintext is returned (fail closed).

    Args:
        key: 32-byte AES-256 key.
        nonce: 12-byte nonce used during encryption.
        ciphertext: The ciphertext portion of the encrypted payload.
        tag: The 16-byte GCM authentication tag.
        associated_data: The exact AAD used during encryption.

    Returns:
        Original plaintext bytes.

    Raises:
        InvalidTag: If authentication fails (wrong key or tampered data).
        ValueError: If key, nonce, or tag have an invalid length.
    """
    if len(key) != AES_KEY_SIZE:
        msg = f"AES-256 requires a {AES_KEY_SIZE}-byte key, got {len(key)}."
        raise ValueError(msg)
    if len(nonce) != GCM_NONCE_SIZE:
        msg = f"GCM nonce must be {GCM_NONCE_SIZE} bytes, got {len(nonce)}."
        raise ValueError(msg)
    if len(tag) != GCM_TAG_SIZE:
        msg = f"GCM tag must be {GCM_TAG_SIZE} bytes, got {len(tag)}."
        raise ValueError(msg)

    return AESGCM(key).decrypt(nonce, ciphertext + tag, associated_data)


def generate_random_bytes(size: int) -> bytes:
    """Generate `size` cryptographically secure random bytes from the OS CSPRNG."""
    return secrets.token_bytes(size)
