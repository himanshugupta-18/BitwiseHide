"""
Authentication API endpoints.

Endpoints:
- POST /auth/register — Register new user
- POST /auth/login — Login with email/username + password
- POST /auth/logout — Stateless logout (client discards tokens)
- POST /auth/refresh — Refresh access token using refresh token
- GET  /auth/me — Get current user profile (requires auth)
- POST /auth/change-password — Change password (requires auth)
- PATCH /auth/me — Update profile (requires auth)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.config import get_settings
from app.core.dependencies import DatabaseDep, get_current_user
from app.models import User
from app.schemas.auth import (
    AuthResponse,
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RefreshTokenRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserCreate,
    UserResponse,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])

CurrentUserDep = Annotated[User, Depends(get_current_user)]


def get_auth_service(db: DatabaseDep) -> AuthService:
    """Provide AuthService with UserRepository."""
    from app.repositories.user_repository import UserRepository
    return AuthService(UserRepository(db))


def _token_response(access_token: str, refresh_token: str) -> TokenResponse:
    """Build a TokenResponse with the access token lifetime from settings."""
    settings = get_settings()
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


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
    request: UserCreate,
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
        tokens=_token_response(access_token, refresh_token),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login with email/username and password",
    responses={
        400: {"description": "Invalid credentials"},
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
        tokens=_token_response(access_token, refresh_token),
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout (stateless)",
    responses={
        200: {"description": "Tokens invalidated client-side"},
    },
)
async def logout() -> MessageResponse:
    """
    Log the client out.

    JWT is stateless — there is no server-side session to revoke, so this
    endpoint does NOT fake revocation. It returns a success response
    instructing the client to discard its stored access and refresh tokens.

    Cookie-based transport is not currently configured, so there are no
    cookies to clear here; if cookie storage is added, delete the auth
    cookies on this response.
    """
    return MessageResponse(
        message="Logged out successfully. Discard your access and refresh tokens."
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    responses={
        400: {"description": "Invalid or expired refresh token"},
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

    return _token_response(access_token, refresh_token)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    responses={
        401: {"description": "Not authenticated"},
    },
)
async def get_me(
    current_user: CurrentUserDep,
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
    current_user: CurrentUserDep,
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
    current_user: CurrentUserDep,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """Update authenticated user's email and/or username."""
    user = await auth_service.update_profile(
        user_id=current_user.id,
        email=request.email,
        username=request.username,
    )
    return UserResponse.model_validate(user)
