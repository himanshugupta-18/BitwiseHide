"""
SQLAlchemy ORM base and shared mixins.

Architecture decisions:
- DeclarativeBase over legacy declarative_base() for SQLAlchemy 2.0+ compatibility.
- TimestampMixin auto-generates created_at/updated_at on every model — eliminates
  manual timestamp management and ensures audit consistency.
- UUIDMixin uses database-generated UUIDs — no application-layer UUID collisions.
- All models inherit from Base, which Alembic auto-discovers for migrations.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Declarative base for all ORM models.

    Alembic's env.py imports this to auto-detect schema changes.
    """


class UUIDMixin:
    """Provides a UUID primary key column."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )


class TimestampMixin:
    """Provides created_at and updated_at audit columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(Base, UUIDMixin, TimestampMixin):
    """
    User account model.

    Stores authentication credentials and profile information.
    Password is stored as bcrypt hash — never plaintext.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        index=True,
        unique=True,
        nullable=False,
    )
    username: Mapped[str] = mapped_column(
        index=True,
        unique=True,
        nullable=False,
    )
    password_hash: Mapped[str] = mapped_column(
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )
    is_verified: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )
