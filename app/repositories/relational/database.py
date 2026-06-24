from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    from app.utils.config import get_settings
    settings = get_settings()
    db = settings.database

    echo = settings.logging.sql_echo
    if db.provider == "sqlite":
        # SQLite uses its own pooling — pool_size/max_overflow are invalid here.
        return create_async_engine(f"sqlite+aiosqlite:///{db.sqlite_path}", echo=echo)

    return create_async_engine(
        db.dsn,
        pool_size=db.pool_size,
        max_overflow=db.max_overflow,
        echo=echo,
    )


def get_session_factory() -> sessionmaker:
    return sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """FastAPI dependency — yields a DB session and commits or rolls back on exit."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
