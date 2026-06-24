from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models.document import Document, DocumentStatus, DocumentVersion


async def create_document(engine: AsyncEngine, doc: Document) -> Document:
    async with engine.begin() as conn:
        await conn.execute(text("""
            INSERT INTO documents (id, user_id, source, mime_type, status, content_hash, created_at)
            VALUES (:id, :user_id, :source, :mime_type, :status, :content_hash, :created_at)
        """), {
            "id": doc.id,
            "user_id": doc.user_id,
            "source": doc.source,
            "mime_type": doc.mime_type,
            "status": doc.status.value,
            "content_hash": None,
            "created_at": doc.created_at,
        })
    return doc


async def update_document_status(
    engine: AsyncEngine,
    document_id: str,
    status: DocumentStatus,
    content_hash: Optional[str] = None,
    version_id: Optional[str] = None,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("""
            UPDATE documents
            SET status = :status,
                content_hash = COALESCE(:content_hash, content_hash),
                current_version_id = COALESCE(:version_id, current_version_id)
            WHERE id = :document_id
        """), {
            "status": status.value,
            "content_hash": content_hash,
            "version_id": version_id,
            "document_id": document_id,
        })


async def get_document(engine: AsyncEngine, document_id: str, user_id: str) -> Optional[Document]:
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, user_id, source, mime_type, status, current_version_id, created_at
            FROM documents WHERE id = :id AND user_id = :user_id
        """), {"id": document_id, "user_id": user_id})
        row = result.fetchone()
    if row is None:
        return None
    return Document(
        id=row.id,
        user_id=row.user_id,
        source=row.source,
        mime_type=row.mime_type,
        status=DocumentStatus(row.status),
        current_version_id=row.current_version_id,
        created_at=row.created_at,
    )


async def list_documents(engine: AsyncEngine, user_id: str) -> List[Document]:
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, user_id, source, mime_type, status, current_version_id, created_at
            FROM documents WHERE user_id = :user_id ORDER BY created_at DESC
        """), {"user_id": user_id})
        rows = result.fetchall()
    return [
        Document(
            id=row.id,
            user_id=row.user_id,
            source=row.source,
            mime_type=row.mime_type,
            status=DocumentStatus(row.status),
            current_version_id=row.current_version_id,
            created_at=row.created_at,
        )
        for row in rows
    ]


async def delete_document(engine: AsyncEngine, document_id: str, user_id: str) -> bool:
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            DELETE FROM documents WHERE id = :id AND user_id = :user_id
        """), {"id": document_id, "user_id": user_id})
    return result.rowcount > 0


async def find_document_by_hash(engine: AsyncEngine, content_hash: str, user_id: str) -> Optional[Document]:
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT id, user_id, source, mime_type, status, current_version_id, created_at
            FROM documents WHERE content_hash = :hash AND user_id = :user_id LIMIT 1
        """), {"hash": content_hash, "user_id": user_id})
        row = result.fetchone()
    if row is None:
        return None
    return Document(
        id=row.id,
        user_id=row.user_id,
        source=row.source,
        mime_type=row.mime_type,
        status=DocumentStatus(row.status),
        current_version_id=row.current_version_id,
        created_at=row.created_at,
    )


async def create_document_version(engine: AsyncEngine, version: DocumentVersion) -> DocumentVersion:
    async with engine.begin() as conn:
        await conn.execute(text("""
            INSERT INTO document_versions (id, document_id, version_no, content_hash, created_at)
            VALUES (:id, :document_id, :version_no, :content_hash, :created_at)
        """), {
            "id": version.id,
            "document_id": version.document_id,
            "version_no": version.version_no,
            "content_hash": version.content_hash,
            "created_at": version.created_at,
        })
    return version
