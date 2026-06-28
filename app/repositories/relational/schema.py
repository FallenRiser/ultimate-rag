import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


async def _add_column_if_missing(engine: AsyncEngine, table: str, column: str, coltype: str) -> None:
    """Idempotent column add for existing DBs (CREATE TABLE IF NOT EXISTS won't add columns to
    a table that already exists). Each runs in its own transaction so a no-op on a fresh DB
    doesn't poison the others. Names are hardcoded, not user input."""
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
        logger.info("Migrated: added column %s.%s", table, column)
    except Exception as exc:
        # Column already exists (SQLite OperationalError / Postgres DuplicateColumn) — expected.
        logger.debug("Column %s.%s already present, skipping migration: %s", table, column, exc)


# Portable DDL: TIMESTAMP (not TIMESTAMPTZ) and no NOW() default so the same
# schema runs on both Postgres and SQLite — every row is inserted with an
# explicit UTC timestamp anyway.
async def create_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                source              TEXT NOT NULL,
                mime_type           TEXT NOT NULL,
                status              TEXT NOT NULL DEFAULT 'pending',
                content_hash        TEXT,
                current_version_id  TEXT,
                created_at          TIMESTAMP
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_versions (
                id           TEXT PRIMARY KEY,
                document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                version_no   INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                created_at   TIMESTAMP
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)"
        ))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                title           TEXT,
                created_at      TIMESTAMP,
                last_active_at  TIMESTAMP
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id)"
        ))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                ordinal     INTEGER NOT NULL DEFAULT 0,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                citations   TEXT,
                created_at  TIMESTAMP
            )
        """))
        # Per-tenant catalog of discovered free-form attribute keys (the convergence catalog).
        # Bounded by distinct keys per tenant (tens), not document count.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS metadata_keys (
                user_id        TEXT NOT NULL,
                key            TEXT NOT NULL,
                value_samples  TEXT,                      -- JSON list, capped (low-cardinality only)
                high_cardinal  INTEGER NOT NULL DEFAULT 0,
                seen_count     INTEGER NOT NULL DEFAULT 0,
                updated_at     TIMESTAMP,
                PRIMARY KEY (user_id, key)
            )
        """))

    # Migrations for DBs created before these columns existed (no-op on fresh DBs).
    await _add_column_if_missing(engine, "chat_messages", "ordinal", "INTEGER NOT NULL DEFAULT 0")
    await _add_column_if_missing(engine, "chat_messages", "citations", "TEXT")

    # Index on ordinal must come after the migration so it works on existing DBs too.
    async with engine.begin() as conn:
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, ordinal)"
        ))
