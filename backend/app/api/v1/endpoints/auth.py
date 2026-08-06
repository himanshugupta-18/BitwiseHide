"""
Authentication API endpoints.

Endpoints:
- POST /auth/register — Register new user
- POST /auth/login — Login with email/username + password
- POST /auth/refresh — Refresh access token using refresh token
- GET  /auth/me — Get current user profile (requires auth)
- POST /auth/change-password — Change password (requires auth)
- PATCH /auth/me — Update profile (requires auth)
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import DatabaseDep
from app.core.exceptions import ValidationError
from app.core.security import decode_token
from app.db.session import get_async_session
from app.schemas.auth import (
    AuthResponse,
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    RefreshTokenRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


# --- Dependency: Current User ---

async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_async_session),
) -> UserResponse:
    """
    Extract and validate current user from Authorization header.

    Expected format: "Bearer <access_token>"

    Raises:
        HTTPException: 401 if token missing, invalid, or expired
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    token = authorization[7:]  # Remove "Bearer "
    try:
        payload = decode_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        ) from e

    if payload.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    from app.repositories.user_repository import UserRepository
    from app.models import User

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(UUID(payload.sub))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return UserResponse.model_validate(user)


def get_auth_service(db: DatabaseDep) -> AuthService:
    """Provide AuthService with UserRepository."""
    from app.repositories.user_repository import UserRepository
    return AuthService(UserRepository(db))


# --- Endpoints ---

@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    responses={
        409: {"description": "Email or username already exists"},
        422: {"description": "Validation error"},
    },
)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """
    Register a new user account.

    - Email must be unique
    - Username must be unique (alphanumeric, underscore, hyphen)
    - Password must be at least 8 characters with upper, lower, digit, special char
    """
    user = await auth_service.register(
        email=request.email,
        username=request.username,
        password=request.password,
    )

    access_token, refresh_token = await auth_service.create_tokens(user)

    return AuthResponse(
        user=UserResponse.model_validate(user),
        tokens=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=1800,  # 30 minutes
        ),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login with email/username and password",
    responses={
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    },
)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """
    Authenticate user and return access + refresh tokens.

    Accepts either email or username as identifier.
    """
    user = await auth_service.authenticate(
        identifier=request.identifier,
        password=request.password,
    )

    access_token, refresh_token = await auth_service.create_tokens(user)

    return AuthResponse(
        user=UserResponse.model_validate(user),
        tokens=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=1800,
        ),
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    responses={
        401: {"description": "Invalid or expired refresh token"},
    },
)
async def refresh(
    request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """
    Get new access token using refresh token.

    Returns new access token and new refresh token (rotation).
    """
    access_token, refresh_token = await auth_service.refresh_tokens(
        refresh_token=request.refresh_token,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    responses={
        401: {"description": "Not authenticated"},
    },
)
async def get_me(
    current_user: UserResponse = Depends(get_current_user),
) -> UserResponse:
    """Get authenticated user's profile."""
    return current_user


@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change password",
    responses={
        400: {"description": "Current password incorrect"},
        401: {"description": "Not authenticated"},
        422: {"description": "New password validation failed"},
    },
)
async def change_password(
    request: ChangePasswordRequest,
    current_user: UserResponse = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Change authenticated user's password."""
    await auth_service.change_password(
        user_id=current_user.id,
        current_password=request.current_password,
        new_password=request.new_password,
    )
    return MessageResponse(message="Password changed successfully")


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update profile",
    responses={
        401: {"description": "Not authenticated"},
        409: {"description": "Email or username already taken"},
    },
)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: UserResponse = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """Update authenticated user's email and/or username."""
    user = await auth_service.update_profile(
        user_id=current_user.id,
        email=request.email,
        username=request.username,
    )
    return UserResponse.model_validate(user)