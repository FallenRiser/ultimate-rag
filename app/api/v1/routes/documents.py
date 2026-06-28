import logging

from fastapi import APIRouter, HTTPException

from app.models.document import Document, DocumentChunk
from app.repositories.graph.factory import create_graph_store
from app.repositories.relational.database import get_engine
from app.repositories.relational.documents import (
    delete_document,
    get_document,
    list_documents,
)
from app.repositories.vector.factory import create_vector_db
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[Document])
async def list_user_documents(user_id: str = "default") -> list[Document]:
    engine = get_engine()
    return await list_documents(engine, user_id)


@router.get("/{document_id}", response_model=Document)
async def get_user_document(document_id: str, user_id: str = "default") -> Document:
    engine = get_engine()
    doc = await get_document(engine, document_id, user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/{document_id}/chunks", response_model=list[DocumentChunk])
async def list_document_chunks(document_id: str, user_id: str = "default") -> list[DocumentChunk]:
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
async def delete_user_document(document_id: str, user_id: str = "default") -> dict:
    engine = get_engine()
    deleted = await delete_document(engine, document_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove vectors for this document (best-effort: the relational row is already gone).
    try:
        await create_vector_db().delete_by_document(document_id, user_id)
    except Exception as exc:
        logger.warning("Vector cleanup failed for document %s: %s", document_id, exc)
        logger.debug("Vector delete_by_document failure (doc=%s)", document_id, exc_info=True)

    # Remove this document's entities/relations/mentions from the knowledge graph (best-effort).
    if get_settings().graph_store.enabled:
        try:
            graph_store = create_graph_store()
            if graph_store is not None:
                await graph_store.delete_by_document(document_id, user_id)
        except Exception as exc:
            logger.warning("Graph cleanup failed for document %s: %s", document_id, exc)
            logger.debug("Graph delete_by_document failure (doc=%s)", document_id, exc_info=True)

    return {"deleted": document_id}


@router.post("/graph/rebuild-communities")
async def rebuild_graph_communities(user_id: str = "default") -> dict:
    """Re-cluster this tenant's knowledge graph and regenerate community reports (the global/thematic
    retrieval layer). Decoupled from ingest — run it on a schedule after cheap ingests, instead of
    re-clustering the whole graph on every document."""
    settings = get_settings()
    if not settings.graph_store.enabled or not settings.enrichment.community_detection.enabled:
        raise HTTPException(
            status_code=400,
            detail="Set graph_store.enabled and enrichment.community_detection.enabled to build communities.",
        )

    from app.services.embeddings.factory import create_embedding_provider
    from app.services.graph.service import GraphRAGService
    from app.services.llm.factory import create_llm_provider

    graph_store = create_graph_store()
    if graph_store is None:
        raise HTTPException(status_code=400, detail="No graph store configured.")

    service = GraphRAGService(graph_store, create_llm_provider(), create_embedding_provider())
    count = await service.rebuild_communities(user_id)
    logger.info("Rebuilt %d communities for user %s", count, user_id)
    return {"communities": count}
