from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


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
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TIMESTAMP
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)"
        ))
