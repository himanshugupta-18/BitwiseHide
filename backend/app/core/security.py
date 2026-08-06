"""
Password hashing and JWT token utilities.

Architecture decisions:
- Argon2id for password hashing — memory-hard, resistant to GPU/ASIC attacks.
- JWT with RS256 (asymmetric) for tokens — public key can be shared for verification
  without exposing signing key. Falls back to HS256 if no keys configured.
- Token payload includes standard claims (sub, exp, iat) plus custom claims.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from pydantic import BaseModel

from app.core.config import get_settings


# --- Password Hashing ---

_password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2id.

    Returns the encoded hash string (includes algorithm, parameters, salt, hash).
    """
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against an Argon2id hash.

    Returns True if valid, False otherwise.
    """
    try:
        _password_hasher.verify(password_hash, password)
        return True
    except Exception:
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


def create_access_token(
    user_id: str,
    email: str,
    username: str,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User's UUID
        email: User's email
        username: User's username
        expires_delta: Custom expiration (default: settings.access_token_expire_minutes)

    Returns:
        Encoded JWT string
    """
    settings = get_settings()
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    iat = datetime.now(UTC)

    payload = {
        "sub": user_id,
        "email": email,
        "username": username,
        "exp": int(expire.timestamp()),
        "iat": int(iat.timestamp()),
        "type": "access",
    }

    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    user_id: str,
    email: str,
    username: str,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: User's UUID
        email: User's email
        username: User's username
        expires_delta: Custom expiration (default: settings.refresh_token_expire_days)

    Returns:
        Encoded JWT string
    """
    settings = get_settings()
    expire = datetime.now(UTC) + (expires_delta or timedelta(days=settings.refresh_token_expire_days))
    iat = datetime.now(UTC)

    payload = {
        "sub": user_id,
        "email": email,
        "username": username,
        "exp": int(expire.timestamp()),
        "iat": int(iat.timestamp()),
        "type": "refresh",
    }

    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT token.

    Args:
        token: Encoded JWT string

    Returns:
        TokenPayload with validated claims

    Raises:
        jwt.ExpiredSignatureError: Token has expired
        jwt.InvalidTokenError: Token is invalid
    """
    settings = get_settings()
    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    return TokenPayload(**payload)


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)