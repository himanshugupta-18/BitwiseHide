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

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_async_session

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
