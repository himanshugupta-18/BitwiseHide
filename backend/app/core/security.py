"""
Password hashing and JWT token utilities.

Architecture decisions:
- bcrypt for password hashing — strong, salted, and widely audited. Passwords are
  never stored in plaintext; only the bcrypt hash is persisted.
- JWT with HS256 (symmetric) for tokens. Access and refresh tokens are signed with
  SEPARATE secrets (settings.jwt_secret_key vs settings.jwt_refresh_secret_key),
  so a leaked access token cannot be used to mint refresh tokens and vice versa.
- Token payload includes standard claims (sub, exp, iat) plus custom claims.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Literal

import bcrypt
import jwt
from pydantic import BaseModel

from app.core.config import get_settings

# bcrypt truncates input at 72 bytes — enforce the limit explicitly so callers
# get a clear error instead of silently truncated passwords.
_BCRYPT_MAX_PASSWORD_BYTES = 72


# --- Password Hashing ---

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plaintext password (max 72 bytes for bcrypt)

    Returns:
        bcrypt hash string (includes algorithm version, salt, cost factor)

    Raises:
        ValueError: If the password exceeds bcrypt's 72-byte limit
    """
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > _BCRYPT_MAX_PASSWORD_BYTES:
        msg = "Password exceeds the 72-byte bcrypt limit."
        raise ValueError(msg)
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against a bcrypt hash.

    Returns True if valid, False otherwise (including hash format errors —
    never raises for malformed stored hashes).
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# --- JWT Token Handling ---

class TokenPayload(BaseModel):
    """Decoded JWT token payload."""

    sub: str  # user ID
    email: str
    username: str
    exp: int  # expiration timestamp
    iat: int  # issued at timestamp
    type: str  # "access" or "refresh"
    jti: str | None = None  # unique token ID (guarantees rotation produces new tokens)


def _encode_token(
    *,
    user_id: str,
    email: str,
    username: str,
    token_type: Literal["access", "refresh"],
    expires_delta: timedelta,
    secret_key: str,
    algorithm: str,
) -> str:
    """Build and sign a JWT with the given secret key."""
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "email": email,
        "username": username,
        "exp": int((now + expires_delta).timestamp()),
        "iat": int(now.timestamp()),
        "type": token_type,
        # Random unique ID — without it, two tokens minted within the same
        # second are byte-identical, breaking refresh-token rotation.
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def create_access_token(
    user_id: str,
    email: str,
    username: str,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT access token signed with the access-token secret.

    Args:
        user_id: User's UUID
        email: User's email
        username: User's username
        expires_delta: Custom expiration (default: settings.access_token_expire_minutes)

    Returns:
        Encoded JWT string
    """
    settings = get_settings()
    return _encode_token(
        user_id=user_id,
        email=email,
        username=username,
        token_type="access",  # noqa: S106 — bandit flags "token" names as passwords
        expires_delta=expires_delta or timedelta(minutes=settings.access_token_expire_minutes),
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(
    user_id: str,
    email: str,
    username: str,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT refresh token signed with the refresh-token secret.

    Uses a separate secret from access tokens — an access token can never be
    accepted as (or converted into) a refresh token.

    Args:
        user_id: User's UUID
        email: User's email
        username: User's username
        expires_delta: Custom expiration (default: settings.refresh_token_expire_days)

    Returns:
        Encoded JWT string
    """
    settings = get_settings()
    return _encode_token(
        user_id=user_id,
        email=email,
        username=username,
        token_type="refresh",  # noqa: S106 — bandit flags "token" names as passwords
        expires_delta=expires_delta or timedelta(days=settings.refresh_token_expire_days),
        secret_key=settings.jwt_refresh_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(
    token: str,
    *,
    token_kind: Literal["access", "refresh"] = "access",  # noqa: S107 — bandit false positive
) -> TokenPayload:
    """
    Decode and validate a JWT token.

    Args:
        token: Encoded JWT string
        token_kind: Which secret to verify against — "access" uses
            settings.jwt_secret_key, "refresh" uses settings.jwt_refresh_secret_key.

    Returns:
        TokenPayload with validated claims

    Raises:
        jwt.ExpiredSignatureError: Token has expired
        jwt.InvalidTokenError: Token is invalid (bad signature, wrong secret, etc.)
    """
    settings = get_settings()
    secret_key = (
        settings.jwt_refresh_secret_key
        if token_kind == "refresh"  # noqa: S105 — bandit false positive
        else settings.jwt_secret_key
    )
    payload = jwt.decode(
        token,
        secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    return TokenPayload(**payload)


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)
