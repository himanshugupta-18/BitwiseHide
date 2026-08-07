import os
import sys
from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

# Add backend to path so `app` package is importable (env.py lives at
# backend/app/db/migrations/env.py — three levels above the backend root).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.core.config import get_settings
from app.models import Base

# this is the Alembic Config object, which provides access to the values within the .ini file in use.
config = context.config

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url_sync)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Use the async database URL from settings
    connectable = create_async_engine(
        settings.database_url,
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio
    asyncio.run(run_migrations_online())
