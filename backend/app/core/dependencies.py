"""
FastAPI dependency injection providers.

Architecture decision: All shared resources (DB sessions, settings, etc.)
are injected via FastAPI's Depends() mechanism. This enables:
- Automatic lifecycle management (sessions open/close per request)
- Easy mocking in tests (override dependencies without touching business code)
- Explicit declaration of what each endpoint needs
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import decode_token
from app.db.session import get_async_session
from app.models import User
from app.repositories.user_repository import UserRepository

# --- Type aliases for cleaner endpoint signatures ---

SettingsDep = Annotated[Settings, Depends(get_settings)]
"""Inject the application settings singleton."""

DatabaseDep = Annotated[AsyncSession, Depends(get_async_session)]
"""Inject an async database session scoped to the current request."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Alias for get_async_session.

    Exists as a named dependency so tests can override `get_db`
    without reaching into the db module internals.
    """
    async for session in get_async_session():
        yield session


async def get_current_user(
    db: DatabaseDep,
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    """
    Resolve the authenticated user from the Authorization header.

    Expected format: "Bearer <access_token>".

    - Extracts the bearer token from the header.
    - Decodes and validates the JWT against the access-token secret.
    - Loads the user from the database and rejects missing/inactive accounts.

    Raises:
        HTTPException: 401 if the header is missing, the token is invalid
        or expired, or the user no longer exists / is inactive.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:]  # Remove "Bearer "
    try:
        payload = decode_token(token, token_kind="access")  # noqa: S106 — bandit false positive
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(UUID(payload.sub))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
