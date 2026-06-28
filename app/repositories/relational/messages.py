import json
from typing import List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models.chat import ChatMessage


def _row_to_message(row) -> ChatMessage:
    return ChatMessage(
        id=row.id,
        session_id=row.session_id,
        role=row.role,
        content=row.content,
        citations=json.loads(row.citations) if row.citations else None,
        created_at=row.created_at,
    )


async def add_message(engine: AsyncEngine, message: ChatMessage, user_id: str) -> ChatMessage:
    """Append a message with a monotonic per-session ordinal (deterministic ordering, unlike
    wall-clock timestamps that can collide within a turn). Citations are stored as JSON."""
    async with engine.begin() as conn:
        # ponytail: MAX(ordinal)+1 isn't atomic across concurrent inserts to the same session;
        # fine since one session's turns are sequential. Add a row lock if that changes.
        row = (await conn.execute(text(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal "
            "FROM chat_messages WHERE session_id = :session_id"
        ), {"session_id": message.session_id})).fetchone()

        await conn.execute(text("""
            INSERT INTO chat_messages (id, session_id, user_id, ordinal, role, content, citations, created_at)
            VALUES (:id, :session_id, :user_id, :ordinal, :role, :content, :citations, :created_at)
        """), {
            "id": message.id,
            "session_id": message.session_id,
            "user_id": user_id,
            "ordinal": row.next_ordinal,
            "role": message.role,
            "content": message.content,
            "citations": json.dumps(message.citations) if message.citations else None,
            "created_at": message.created_at,
        })
    return message


async def list_recent_messages(
    engine: AsyncEngine, session_id: str, user_id: str, limit: int
) -> List[ChatMessage]:
    """Most recent `limit` messages for a session, returned in chronological order."""
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, session_id, role, content, citations, created_at
            FROM chat_messages
            WHERE session_id = :session_id AND user_id = :user_id
            ORDER BY ordinal DESC
            LIMIT :limit
        """), {"session_id": session_id, "user_id": user_id, "limit": limit})
        rows = result.fetchall()

    messages = [_row_to_message(row) for row in rows]
    messages.reverse()
    return messages


async def list_messages(
    engine: AsyncEngine, session_id: str, user_id: str, limit: int = 50, offset: int = 0
) -> List[ChatMessage]:
    """A session's messages in chronological order (oldest first), paginated — for rendering
    a conversation. (list_recent_messages instead loads the latest N for agent history.)"""
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, session_id, role, content, citations, created_at
            FROM chat_messages
            WHERE session_id = :session_id AND user_id = :user_id
            ORDER BY ordinal ASC
            LIMIT :limit OFFSET :offset
        """), {"session_id": session_id, "user_id": user_id, "limit": limit, "offset": offset})
        rows = result.fetchall()
    return [_row_to_message(row) for row in rows]


async def count_messages(engine: AsyncEngine, session_id: str, user_id: str) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT COUNT(*) AS n FROM chat_messages WHERE session_id = :session_id AND user_id = :user_id"
        ), {"session_id": session_id, "user_id": user_id})
        return result.fetchone().n
