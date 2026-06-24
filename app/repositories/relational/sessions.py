from datetime import datetime
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models.chat import ChatSession


async def create_session(engine: AsyncEngine, session: ChatSession) -> ChatSession:
    async with engine.begin() as conn:
        await conn.execute(text("""
            INSERT INTO chat_sessions (id, user_id, title, created_at, last_active_at)
            VALUES (:id, :user_id, :title, :created_at, :last_active_at)
        """), {
            "id": session.id,
            "user_id": session.user_id,
            "title": session.title,
            "created_at": session.created_at,
            "last_active_at": session.last_active_at,
        })
    return session


async def touch_session(engine: AsyncEngine, session_id: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("""
            UPDATE chat_sessions SET last_active_at = :now WHERE id = :id
        """), {"id": session_id, "now": datetime.utcnow()})


async def get_session(engine: AsyncEngine, session_id: str, user_id: str) -> Optional[ChatSession]:
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, user_id, title, created_at, last_active_at
            FROM chat_sessions WHERE id = :id AND user_id = :user_id
        """), {"id": session_id, "user_id": user_id})
        row = result.fetchone()
    if row is None:
        return None
    return ChatSession(
        id=row.id,
        user_id=row.user_id,
        title=row.title,
        created_at=row.created_at,
        last_active_at=row.last_active_at,
    )


async def list_sessions(engine: AsyncEngine, user_id: str) -> List[ChatSession]:
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, user_id, title, created_at, last_active_at
            FROM chat_sessions WHERE user_id = :user_id ORDER BY last_active_at DESC
        """), {"user_id": user_id})
        rows = result.fetchall()
    return [
        ChatSession(
            id=row.id, user_id=row.user_id, title=row.title,
            created_at=row.created_at, last_active_at=row.last_active_at,
        )
        for row in rows
    ]


async def delete_session(engine: AsyncEngine, session_id: str, user_id: str) -> bool:
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            DELETE FROM chat_sessions WHERE id = :id AND user_id = :user_id
        """), {"id": session_id, "user_id": user_id})
    return result.rowcount > 0
