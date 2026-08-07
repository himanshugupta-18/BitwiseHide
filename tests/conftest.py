"""
Configuration for pytest.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.session import get_async_session
from app.models import Base
from app.main import create_app
from app.models import User

# Test database URL - using SQLite in memory for fast tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="function")
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with async_session() as session:
        yield session


@pytest.fixture(scope="function")
def override_settings() -> Settings:
    """Override settings for testing."""
    return Settings(
        app_env="testing",
        debug=True,
        database_url=TEST_DATABASE_URL,
        jwt_secret_key="test-secret-key-for-testing-only",
        jwt_refresh_secret_key="test-refresh-secret-key-for-testing-only",
        jwt_algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7,
    )


@pytest.fixture(scope="function")
def app(override_settings, test_session):
    """Create test FastAPI application."""
    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_settings] = lambda: override_settings
    app.dependency_overrides[get_async_session] = lambda: test_session

    yield app

    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="function")
async def test_user(test_session) -> User:
    """Create a test user."""
    from app.core.security import hash_password

    user = User(
        email="test@example.com",
        username="testuser",
        password_hash=hash_password("TestPass123!"),
    )
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)
    return user


@pytest.fixture(scope="function")
async def auth_headers(test_user) -> dict[str, str]:
    """Create authorization headers for test user."""
    from app.core.security import create_access_token

    access_token = create_access_token(
        user_id=str(test_user.id),
        email=test_user.email,
        username=test_user.username,
    )
    return {"Authorization": f"Bearer {access_token}"}