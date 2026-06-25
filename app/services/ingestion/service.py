import io
import logging
from typing import Optional, Tuple

from app.models.document import Document, DocumentStatus, DocumentVersion
from app.models.ingestion import IngestionResponse
from app.repositories.relational.database import get_engine
from app.repositories.relational.documents import (
    create_document,
    create_document_version,
    find_document_by_hash,
    update_document_status,
)
from app.repositories.vector.factory import create_vector_db
from app.services.chunking.factory import create_chunker
from app.services.embeddings.factory import create_embedding_provider
from app.services.parsing.factory import create_parser
from app.utils.config import get_settings
from app.utils.hashing import sha256_bytes, sha256_text

logger = logging.getLogger(__name__)


class IngestionService:
    """In-process ingestion. `register` records the document; `process` runs the
    parse → chunk → embed → index pipeline. The route runs `process` either in a
    background task (async endpoint) or inline (sync endpoint)."""

    async def register(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        user_id: str,
    ) -> Tuple[Document, Optional[IngestionResponse]]:
        """Validate, dedup, and create the document row. Returns (document, dedup_response).
        If dedup_response is not None, the caller should return it and skip processing."""
        settings = get_settings()
        engine = get_engine()

        max_bytes = settings.ingestion.max_file_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(f"File exceeds max size of {settings.ingestion.max_file_mb} MB")

        content_hash = sha256_bytes(content)

        if settings.ingestion.dedup_by_hash:
            existing = await find_document_by_hash(engine, content_hash, user_id)
            if existing:
                logger.info("Duplicate document for user %s (hash %s)", user_id, content_hash[:8])
                return existing, IngestionResponse(
                    document_id=existing.id, task_id="dedup", status="duplicate"
                )

        doc = Document(user_id=user_id, source=filename, mime_type=mime_type)
        await create_document(engine, doc)
        return doc, None

    async def process(
        self,
        document_id: str,
        user_id: str,
        content: bytes,
        filename: str,
        mime_type: str,
        content_hash: str,
        extra_metadata: Optional[dict] = None,
        parser_options: Optional[dict] = None,
    ) -> DocumentStatus:
        """Full pipeline: parse → chunk → embed → upsert vector DB. Returns final status."""
        engine = get_engine()
        vector_db = create_vector_db()
        embedding_provider = create_embedding_provider()
        parser = create_parser()
        chunker = create_chunker()

        try:
            await update_document_status(engine, document_id, DocumentStatus.processing)

            settings = get_settings()

            logger.info("Parsing %s (%s)", filename, mime_type)
            parsed = await parser.parse(io.BytesIO(content), mime_type, filename, options=parser_options)
            logger.debug("Parsed %s: %d chars, %d pages", filename, len(parsed.text), len(parsed.pages))

            version = DocumentVersion(
                document_id=document_id,
                version_no=1,
                content_hash=content_hash or sha256_text(parsed.text),
            )
            await create_document_version(engine, version)

            logger.info("Chunking %s (strategy=%s)", filename, settings.chunking.strategy)
            chunks = await chunker.chunk(parsed.text, document_id, version.id, parsed.metadata)
            if not chunks:
                logger.warning("Document %s produced no chunks after parsing", document_id)
                await update_document_status(engine, document_id, DocumentStatus.failed)
                return DocumentStatus.failed
            logger.info("Chunked %s into %d chunks", filename, len(chunks))
            for chunk in chunks:
                logger.debug(
                    "  chunk %d (page=%s, tokens=%d): %.80s",
                    chunk.ordinal, chunk.page, chunk.token_count, chunk.text.replace("\n", " "),
                )

            # Document-level metadata (title/author/doc_type/topics), stamped on every
            # chunk so it's filterable at query time and visible in citations.
            doc_metadata = {}
            if settings.enrichment.metadata_extraction.enabled:
                try:
                    doc_metadata = await _extract_document_metadata(parsed.text)
                    logger.info("Extracted metadata for %s: %s", document_id, doc_metadata)
                except Exception as exc:
                    logger.warning("Metadata extraction failed for %s: %s", document_id, exc)

            batch_size = settings.embeddings.batch_size
            logger.info(
                "Embedding %d chunks (provider=%s, model=%s)",
                len(chunks), settings.embeddings.provider, settings.embeddings.model,
            )
            all_vectors = []
            for i in range(0, len(chunks), batch_size):
                batch_texts = [c.text for c in chunks[i: i + batch_size]]
                logger.debug("Embedding batch %d-%d of %d", i, i + len(batch_texts), len(chunks))
                all_vectors.extend(await embedding_provider.embed_documents(batch_texts))

            chunk_dicts = [
                {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "dense_vector": vector,
                    "metadata": {
                        "document_id": document_id,
                        "version_id": version.id,
                        "ordinal": chunk.ordinal,
                        "page": chunk.page,
                        **chunk.metadata,
                        **doc_metadata,                 # LLM-extracted, document-level
                        **(extra_metadata or {}),       # caller-supplied wins on conflict
                    },
                }
                for chunk, vector in zip(chunks, all_vectors)
            ]
            logger.info("Indexing %d chunks into %s", len(chunk_dicts), settings.vector_store.provider)
            await vector_db.insert(chunk_dicts, user_id)

            # Populate the knowledge graph whenever a graph store is configured —
            # enabling the graph store is the intent to fill it.
            if settings.graph_store.enabled:
                await _extract_and_store_entities(chunks, document_id, user_id)

            await update_document_status(
                engine, document_id, DocumentStatus.ready,
                content_hash=content_hash, version_id=version.id,
            )
            logger.info("Ingested document %s (%d chunks) for user %s", document_id, len(chunks), user_id)
            return DocumentStatus.ready

        except Exception as exc:
            logger.exception("Ingestion failed for document %s: %s", document_id, exc)
            await update_document_status(engine, document_id, DocumentStatus.failed)
            return DocumentStatus.failed


async def _extract_document_metadata(text: str) -> dict:
    """One LLM call per document → title/author/doc_type/topics. Scalar fields are
    filterable at query time; topics is joined to a string for display/substring use."""
    from pydantic import BaseModel as _BaseModel

    from app.services.llm.factory import create_llm_provider

    class _DocMeta(_BaseModel):
        title: str = ""
        author: str = ""
        doc_type: str = ""          # e.g. report, 10-Q, article, email, contract
        topics: list[str] = []

    llm = create_llm_provider()
    result: _DocMeta = await llm.structured_output(
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract document-level metadata from the text. Provide title, author, "
                    "doc_type (report, 10-Q, article, email, contract, etc.), and up to 5 topics. "
                    "Use empty strings / empty list when unknown. Do not invent values."
                ),
            },
            {"role": "user", "content": text[:3000]},
        ],
        schema=_DocMeta,
    )

    metadata = {}
    if result.title:
        metadata["title"] = result.title
    if result.author:
        metadata["author"] = result.author
    if result.doc_type:
        metadata["doc_type"] = result.doc_type
    if result.topics:
        metadata["topics"] = ", ".join(result.topics)  # flat string (stores need scalar values)
    return metadata


async def _extract_and_store_entities(chunks, document_id: str, user_id: str) -> None:
    """Use the LLM to extract named entities from chunks and store them in the graph."""
    from typing import List as _List

    from pydantic import BaseModel as _BaseModel

    from app.repositories.graph.factory import create_graph_store
    from app.services.llm.factory import create_llm_provider

    class _Entity(_BaseModel):
        id: str
        name: str
        type: str = "Entity"

    class _EntityList(_BaseModel):
        entities: _List[_Entity] = []

    graph_store = create_graph_store()
    if graph_store is None:
        return

    llm = create_llm_provider()

    for chunk in chunks[:20]:  # cap at 20 chunks to limit LLM cost per document
        try:
            result: _EntityList = await llm.structured_output(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract named entities (people, organisations, places, products, concepts) "
                            "from the text. For each entity, provide a stable id (snake_case), display name, "
                            "and type. Return JSON with an 'entities' array."
                        ),
                    },
                    {"role": "user", "content": chunk.text[:1000]},
                ],
                schema=_EntityList,
            )
        except Exception as exc:
            logger.warning("Entity extraction failed for chunk %s: %s", chunk.id, exc)
            continue

        for entity in result.entities:
            await graph_store.upsert_entity(
                {
                    "id": f"{document_id}::{entity.id}",
                    "name": entity.name,
                    "type": entity.type,
                    "document_id": document_id,
                    "chunk_id": chunk.id,
                    "properties": {},
                },
                user_id,
            )
