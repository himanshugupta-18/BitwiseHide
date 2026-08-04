"""
Async database engine and session factory.

Architecture decisions:
- asyncpg driver for non-blocking I/O in the FastAPI event loop.
- pool_pre_ping=True verifies connection liveness before checkout,
  preventing "stale connection" errors after DB restarts.
- pool_recycle rotates connections hourly to avoid hitting PostgreSQL
  max-connection-age limits.
- Session factory is a callable — FastAPI's Depends() calls it per-request
  to get a scoped session that auto-closes after the request lifecycle.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level references — initialized by init_db(), torn down by close_db()
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_db(settings: Settings) -> None:
    """
    Create the async engine and session factory.

    Called once during application lifespan startup.
    Uses module-level state so the engine is shared across all requests.
    """
    global _engine, _session_factory  # noqa: PLW0603

    _engine = create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
        echo=settings.db_echo,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    logger.info(
        "database_initialized",
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )


async def close_db() -> None:
    """
    Dispose the engine and release all pooled connections.

    Called once during application lifespan shutdown.
    """
    global _engine, _session_factory  # noqa: PLW0603

    if _engine is not None:
        await _engine.dispose()
        logger.info("database_connections_closed")

    _engine = None
    _session_factory = None


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an AsyncSession for a single request lifecycle.

    Usage as a FastAPI dependency:
        async def endpoint(db: AsyncSession = Depends(get_async_session)):
            ...

    The session is committed on success, rolled back on exception,
    and always closed when the request completes.
    """
    if _session_factory is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_engine() -> AsyncEngine:
    """Return the active engine (used by Alembic and health checks)."""
    if _engine is None:
        msg = "Database not initialized. Call init_db() first."
        raise RuntimeError(msg)
    return _engine
