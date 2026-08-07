"""
Authentication API schemas — Pydantic models for request/response validation.

Architecture decision: Separate schemas for requests (input) and responses (output).
Input schemas validate incoming data; output schemas control what's exposed to clients.

Password policy (enforced both here and in the service layer for defense in depth):
- minimum 8 characters
- at least one uppercase letter
- at least one lowercase letter
- at least one number
- at least one special character
- no more than 72 bytes (bcrypt hard limit)
"""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# Characters considered "special" for password strength purposes.
_SPECIAL_CHARACTERS = re.compile(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]")
# bcrypt cannot process passwords longer than 72 bytes.
_BCRYPT_MAX_PASSWORD_BYTES = 72


def validate_password_strength(value: str) -> str:
    """Enforce the shared password policy. Raises ValueError with a clear message."""
    if len(value) < 8:
        raise ValueError("Password must be at least 8 characters long")
    if not any(char.isupper() for char in value):
        raise ValueError("Password must contain at least one uppercase letter")
    if not any(char.islower() for char in value):
        raise ValueError("Password must contain at least one lowercase letter")
    if not any(char.isdigit() for char in value):
        raise ValueError("Password must contain at least one number")
    if not _SPECIAL_CHARACTERS.search(value):
        raise ValueError("Password must contain at least one special character")
    if len(value.encode("utf-8")) > _BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError("Password must be at most 72 bytes (bcrypt limit)")
    return value


# --- Request Schemas ---

class UserCreate(BaseModel):
    """User registration request."""

    email: EmailStr = Field(..., description="User's email address")
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Username (alphanumeric, underscore, hyphen only)",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (min 8 chars, upper, lower, digit, special)",
    )

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        return validate_password_strength(value)


class LoginRequest(BaseModel):
    """User login request."""

    identifier: str = Field(
        ...,
        min_length=1,
        description="Email or username",
    )
    password: str = Field(..., min_length=1, description="Password")


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str = Field(..., min_length=1, description="Refresh token")


class ChangePasswordRequest(BaseModel):
    """Password change request."""

    current_password: str = Field(..., min_length=1, description="Current password")
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password (min 8 chars, upper, lower, digit, special)",
    )

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: str) -> str:
        return validate_password_strength(value)


class UpdateProfileRequest(BaseModel):
    """Profile update request."""

    email: EmailStr | None = Field(None, description="New email address")
    username: str | None = Field(
        None,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="New username",
    )


# --- Response Schemas ---

class UserResponse(BaseModel):
    """User profile response (safe fields only)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    username: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    """Token pair response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 — bandit flags "token" names as passwords
    expires_in: int  # access token lifetime in seconds


class AuthResponse(BaseModel):
    """Full authentication response with user and tokens."""

    user: UserResponse
    tokens: TokenResponse


class MessageResponse(BaseModel):
    """Generic success message response."""

    message: str


# --- Internal Schemas ---

class CurrentUser(BaseModel):
    """Current authenticated user (from token)."""

    id: UUID
    email: str
    username: str
