from typing import List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models.chat import ChatMessage


async def add_message(engine: AsyncEngine, message: ChatMessage, user_id: str) -> ChatMessage:
    async with engine.begin() as conn:
        await conn.execute(text("""
            INSERT INTO chat_messages (id, session_id, user_id, role, content, created_at)
            VALUES (:id, :session_id, :user_id, :role, :content, :created_at)
        """), {
            "id": message.id,
            "session_id": message.session_id,
            "user_id": user_id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at,
        })
    return message


async def list_recent_messages(
    engine: AsyncEngine, session_id: str, user_id: str, limit: int
) -> List[ChatMessage]:
    """Most recent `limit` messages for a session, returned in chronological order."""
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, session_id, role, content, created_at
            FROM chat_messages
            WHERE session_id = :session_id AND user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"session_id": session_id, "user_id": user_id, "limit": limit})
        rows = result.fetchall()

    messages = [
        ChatMessage(
            id=row.id,
            session_id=row.session_id,
            role=row.role,
            content=row.content,
            created_at=row.created_at,
        )
        for row in rows
    ]
    messages.reverse()
    return messages
