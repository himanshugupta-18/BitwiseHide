"""
User repository — data access layer for User model.

Architecture decision: Repository pattern isolates database queries from business logic.
Services depend on repository interfaces, enabling easy testing with mocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User


class UserRepository:
    """
    Data access operations for User entities.

    All methods are async and use SQLAlchemy 2.0 select() API.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        email: str,
        username: str,
        password_hash: str,
    ) -> User:
        """
        Create a new user.

        Args:
            email: User's email (unique)
            username: User's username (unique)
            password_hash: bcrypt hashed password

        Returns:
            Created User entity
        """
        user = User(
            email=email.lower(),
            username=username,
            password_hash=password_hash,
        )
        self._session.add(user)
        await self._session.flush()
        await self._session.refresh(user)
        return user

    async def get_by_id(self, user_id: UUID) -> User | None:
        """Get user by UUID."""
        stmt = select(User).where(User.id == user_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email (case-insensitive)."""
        stmt = select(User).where(User.email == email.lower())
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username."""
        stmt = select(User).where(User.username == username)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email_or_username(self, identifier: str) -> User | None:
        """
        Get user by email or username.

        Used for login where user can provide either.
        """
        stmt = select(User).where(
            (User.email == identifier.lower()) | (User.username == identifier)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists_by_email(self, email: str) -> bool:
        """Check if email is already registered."""
        stmt = select(User.id).where(User.email == email.lower())
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def exists_by_username(self, username: str) -> bool:
        """Check if username is already taken."""
        stmt = select(User.id).where(User.username == username)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def list_users(self, *, offset: int = 0, limit: int = 100) -> Sequence[User]:
        """List users with pagination."""
        stmt = select(User).offset(offset).limit(limit).order_by(User.created_at.desc())
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def update(self, user: User) -> User:
        """Update user (e.g., password change, verification status)."""
        await self._session.flush()
        await self._session.refresh(user)
        return user

    async def delete(self, user: User) -> None:
        """Delete user (soft delete via is_active=False recommended instead)."""
        await self._session.delete(user)
        await self._session.flush()