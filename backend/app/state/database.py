"""
Database connection and initialization.

Phase 0: SQLite via aiosqlite for async support.
Phase 1: Migrate to PostgreSQL with asyncpg.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.state.models import Base

logger = structlog.get_logger(__name__)

# Module-level engine and session factory
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_database(db_url: str | None = None) -> None:
    """
    Initialize the database engine and create tables.

    Call this once at application startup (in main.py).
    `db_url` optionally overrides the configured URL (used by offline tools
    such as the RL dataset exporter's `--db` flag). Must be a full SQLAlchemy
    async URL (e.g. "sqlite+aiosqlite:///C:/path/agent.db").
    """
    global _engine, _session_factory

    settings = get_settings()
    if db_url is None:
        db_url = settings.db_url

    # Ensure the data directory exists for SQLite
    if "sqlite" in db_url:
        db_path = db_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("database_initializing", url=db_url)

    _engine = create_async_engine(
        db_url,
        echo=False,  # Set to True for SQL debugging
        future=True,
    )

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create all tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("database_initialized")


async def get_session() -> AsyncSession:
    """
    Get a new async database session.

    Usage:
        session = await get_session()
        async with session.begin():
            ...
    """
    global _session_factory
    if _session_factory is None:
        await init_database()
    assert _session_factory is not None
    return _session_factory()


async def close_database() -> None:
    """Close the database engine."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database_closed")
