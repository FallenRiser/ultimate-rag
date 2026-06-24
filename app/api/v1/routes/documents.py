from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.document import Document, DocumentChunk
from app.repositories.relational.database import get_engine
from app.repositories.relational.documents import (
    delete_document,
    get_document,
    list_documents,
)
from app.repositories.vector.factory import create_vector_db

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[Document])
async def list_user_documents(
    user_id: str = Depends(get_current_user),
) -> list[Document]:
    engine = get_engine()
    return await list_documents(engine, user_id)


@router.get("/{document_id}", response_model=Document)
async def get_user_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
) -> Document:
    engine = get_engine()
    doc = await get_document(engine, document_id, user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/{document_id}/chunks", response_model=list[DocumentChunk])
async def list_document_chunks(
    document_id: str,
    user_id: str = Depends(get_current_user),
) -> list[DocumentChunk]:
    # Confirm the document belongs to this user before exposing its chunks.
    doc = await get_document(get_engine(), document_id, user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    rows = await create_vector_db().list_by_document(document_id, user_id)
    chunks = [
        DocumentChunk(
            chunk_id=str(row["id"]),
            document_id=row["payload"].get("metadata", {}).get("document_id", document_id),
            ordinal=row["payload"].get("metadata", {}).get("ordinal"),
            page=row["payload"].get("metadata", {}).get("page"),
            text=row["payload"].get("text", ""),
            metadata=row["payload"].get("metadata", {}) or {},
        )
        for row in rows
    ]
    # Return in document order; chunks without an ordinal sort last.
    chunks.sort(key=lambda c: c.ordinal if c.ordinal is not None else 1_000_000)
    return chunks


@router.delete("/{document_id}")
async def delete_user_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    engine = get_engine()
    deleted = await delete_document(engine, document_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove vectors for this document (best-effort; don't fail the request if vector DB is down)
    try:
        vector_db = create_vector_db()
        await vector_db.delete_by_document(document_id, user_id)
    except Exception:
        pass

    return {"deleted": document_id}
