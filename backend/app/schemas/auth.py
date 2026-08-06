"""
Authentication API schemas — Pydantic models for request/response validation.

Architecture decision: Separate schemas for requests (input) and responses (output).
Input schemas validate incoming data; output schemas control what's exposed to clients.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# --- Request Schemas ---

class RegisterRequest(BaseModel):
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

    id: UUID
    email: EmailStr
    username: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token pair response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
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