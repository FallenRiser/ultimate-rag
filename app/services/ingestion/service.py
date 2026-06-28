import io
import logging
from typing import Optional, Tuple

from sqlalchemy.exc import SQLAlchemyError

from app.models.document import Document, DocumentStatus, DocumentVersion
from app.models.ingestion import (
    ChunkAttributes,
    DocumentMetadata,
    ExtractedGraph,
    IngestionResponse,
)
from app.prompts.ingestion import (
    GRAPH_EXTRACTION_SYSTEM,
    catalog_hint,
    document_metadata_system,
    section_metadata_system,
)
from app.repositories.graph.factory import create_graph_store
from app.repositories.relational.database import get_engine
from app.repositories.relational.metadata_catalog import get_catalog, record_keys
from app.repositories.relational.documents import (
    create_document,
    create_document_version,
    find_document_by_hash,
    update_document_status,
)
from app.repositories.vector.factory import create_vector_db
from app.services.chunking.factory import create_chunker
from app.services.embeddings.factory import create_embedding_provider
from app.services.llm.factory import create_llm_provider
from app.services.parsing.factory import create_parser
from app.utils.config import get_settings
from app.utils.hashing import sha256_bytes, sha256_text
from app.utils.text import cap_list, cap_text

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
                logger.info(
                    "Duplicate document for user %s (hash %s)",
                    user_id, cap_text(content_hash, settings.logging.preview_chars),
                )
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
                preview = cap_text(chunk.text.replace("\n", " "), settings.logging.preview_chars)
                logger.debug(
                    "  chunk %d (page=%s, tokens=%d): %s",
                    chunk.ordinal, chunk.page, chunk.token_count, preview,
                )

            # Document-level metadata (core fields + free-form attributes), stamped on every
            # chunk so it's filterable at query time and visible in citations.
            doc_metadata = {}
            if settings.enrichment.metadata_extraction.enabled:
                # Non-breaking: a document still ingests if metadata extraction fails.
                try:
                    doc_metadata = await _extract_document_metadata(parsed.text, user_id, engine)
                    logger.info("Extracted metadata for %s: %s", document_id, doc_metadata)
                except Exception as exc:
                    logger.warning("Metadata extraction failed for %s; ingesting without it: %s", document_id, exc)
                    logger.debug("Metadata extraction failure (doc=%s)", document_id, exc_info=True)

            # Section-level (per-chunk) attributes — merged into each chunk's own metadata.
            if settings.enrichment.section_metadata.enabled:
                await _extract_section_metadata_into_chunks(chunks, user_id, engine)

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
                await _extract_and_store_graph(chunks, document_id, user_id, embedding_provider)

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


# Core fields stored as-is for display; doc_type/language are also lowercased so they filter
# reliably (exact-match filtering is case-sensitive). Reserved keys can't be overwritten by
# a free-form attribute.
_CORE_FIELDS = ("title", "author", "doc_type", "summary", "language", "created_date")
_NORMALIZED_CORE = {"doc_type", "language"}
_RESERVED_KEYS = {"user_id", "document_id", "version_id", "ordinal", "page", *_CORE_FIELDS}


def _normalize_attributes(pairs) -> dict:
    """Free-form pairs → flat {key: value}, with keys/values normalized for exact-match
    filtering and reserved/system keys dropped."""
    out: dict = {}
    for pair in pairs:
        key = pair.key.strip().lower().replace(" ", "_")
        value = pair.value.strip().lower()
        if key and value and key not in _RESERVED_KEYS:
            out[key] = value
    return out


async def _extract_document_metadata(text: str, user_id: str, engine) -> dict:
    """One LLM call per document → core fields + free-form attributes (see DocumentMetadata).
    Existing tenant keys are fed in so the attribute key-space converges; newly seen keys are
    recorded back into the catalog. Input length governed by metadata_extraction.max_chars."""
    cfg = get_settings().enrichment.metadata_extraction
    llm_input = cap_text(text, cfg.max_chars)

    # Non-breaking: extract without key hints if the catalog can't be read.
    catalog = []
    try:
        catalog = await get_catalog(engine, user_id)
    except SQLAlchemyError as exc:
        logger.warning("Catalog read failed for %s; extracting without key hints: %s", user_id, exc)
        logger.debug("Catalog read DB failure (user=%s)", user_id, exc_info=True)
    except Exception as exc:
        logger.warning("Unexpected error reading catalog for %s; extracting without hints: %s", user_id, exc)
        logger.debug("Unexpected catalog read failure (user=%s)", user_id, exc_info=True)
    logger.debug("Metadata extraction for %s: %d known keys, input %d chars", user_id, len(catalog), len(llm_input))

    llm = create_llm_provider()
    result: DocumentMetadata = await llm.structured_output(
        messages=[
            {"role": "system", "content": document_metadata_system(catalog_hint(catalog, cfg.attribute_hints))},
            {"role": "user", "content": llm_input},
        ],
        schema=DocumentMetadata,
    )

    metadata: dict = {}
    for field in _CORE_FIELDS:
        value = getattr(result, field)
        if value:
            metadata[field] = value.lower() if field in _NORMALIZED_CORE else value

    attributes = _normalize_attributes(result.attributes)
    metadata.update(attributes)   # flat, top-level → filterable across all vector stores
    logger.debug("Extracted %d core field(s) and %d attribute(s) for %s",
                 len(metadata) - len(attributes), len(attributes), user_id)
    if attributes:
        # Non-breaking: don't discard the metadata we already extracted over a catalog write.
        try:
            await record_keys(engine, user_id, attributes)
        except SQLAlchemyError as exc:
            logger.warning("Catalog update failed for %s: %s", user_id, exc)
            logger.debug("Catalog write DB failure (user=%s, keys=%s)", user_id, list(attributes), exc_info=True)
        except Exception as exc:
            logger.warning("Unexpected error updating catalog for %s: %s", user_id, exc)
            logger.debug("Unexpected catalog write failure (user=%s, keys=%s)", user_id, list(attributes), exc_info=True)
    return metadata


async def _extract_section_metadata_into_chunks(chunks, user_id: str, engine) -> None:
    """Per-chunk attribute extraction (Vectorize's chunk_metadata). Each chunk's attributes are
    normalized, merged into its own metadata, and recorded into the catalog. One LLM call per
    chunk — capped by enrichment.section_metadata.{max_chunks,max_chars}; failures are skipped."""
    cfg = get_settings().enrichment.section_metadata
    # Non-breaking: extract without key hints if the catalog can't be read.
    catalog = []
    try:
        catalog = await get_catalog(engine, user_id)
    except SQLAlchemyError as exc:
        logger.warning("Catalog read failed for %s; section extraction without key hints: %s", user_id, exc)
        logger.debug("Catalog read DB failure (user=%s)", user_id, exc_info=True)
    except Exception as exc:
        logger.warning("Unexpected error reading catalog for %s; section extraction without hints: %s", user_id, exc)
        logger.debug("Unexpected catalog read failure (user=%s)", user_id, exc_info=True)
    system_prompt = section_metadata_system(catalog_hint(catalog, cfg.attribute_hints))
    llm = create_llm_provider()

    targets = cap_list(chunks, cfg.max_chunks)
    logger.info("Extracting section metadata from %d chunks for user %s", len(targets), user_id)
    for chunk in targets:
        try:
            result: ChunkAttributes = await llm.structured_output(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": cap_text(chunk.text, cfg.max_chars)},
                ],
                schema=ChunkAttributes,
            )
        except Exception as exc:
            # Non-breaking: skip this chunk's section metadata, keep ingesting the rest.
            logger.warning("Section metadata extraction failed for chunk %s: %s", chunk.id, exc)
            logger.debug("Section extraction failure (chunk=%s)", chunk.id, exc_info=True)
            continue

        attributes = _normalize_attributes(result.attributes)
        if attributes:
            chunk.metadata.update(attributes)
            logger.debug("Chunk %s section attributes: %s", chunk.id, attributes)
            try:
                await record_keys(engine, user_id, attributes)
            except SQLAlchemyError as exc:
                logger.warning("Catalog update failed for chunk %s: %s", chunk.id, exc)
                logger.debug("Catalog write DB failure (chunk=%s, keys=%s)", chunk.id, list(attributes), exc_info=True)
            except Exception as exc:
                logger.warning("Unexpected error updating catalog for chunk %s: %s", chunk.id, exc)
                logger.debug("Unexpected catalog write failure (chunk=%s)", chunk.id, exc_info=True)


def _entity_key(name: str) -> str:
    """Normalised graph node id. The same name in different documents maps to the same key,
    so entities merge across the tenant's documents into one node (the heart of GraphRAG)."""
    return name.strip().lower().replace(" ", "_")


async def _extract_and_store_graph(chunks, document_id: str, user_id: str, embedding_provider) -> None:
    """Per chunk, the LLM extracts entities + relations (with descriptions). Entities merge across
    documents by normalized name and are embedded for semantic query matching; relations become
    edges. Optionally summarises merged descriptions and rebuilds communities (config-gated).
    Scope governed by enrichment.entity_extraction (max_chunks/max_chars; 0 = no limit)."""
    graph_store = create_graph_store()
    if graph_store is None:
        return

    settings = get_settings()
    cfg = settings.enrichment.entity_extraction
    llm = create_llm_provider()

    entity_records: list = []     # one per (entity, chunk) mention — the store merges them
    relation_records: list = []
    rep_text: dict = {}           # entity key -> best "name: description" text for embedding

    for chunk in cap_list(chunks, cfg.max_chunks):
        try:
            result: ExtractedGraph = await llm.structured_output(
                messages=[
                    {"role": "system", "content": GRAPH_EXTRACTION_SYSTEM},
                    {"role": "user", "content": cap_text(chunk.text, cfg.max_chars)},
                ],
                schema=ExtractedGraph,
            )
        except Exception as exc:
            # Non-breaking: skip this chunk's graph, keep going.
            logger.warning("Graph extraction failed for chunk %s: %s", chunk.id, exc)
            logger.debug("Graph extraction failure (chunk=%s)", chunk.id, exc_info=True)
            continue

        keys: dict = {}  # name -> node key for this chunk; also gates which relations we keep
        for entity in result.entities:
            key = _entity_key(entity.name)
            if not key:
                continue
            keys[entity.name] = key
            entity_records.append({
                "id": key, "name": entity.name, "type": entity.type,
                "description": entity.description,
                "document_id": document_id, "chunk_id": chunk.id,
            })
            text = f"{entity.name}: {entity.description}".strip(": ").strip()
            if len(text) > len(rep_text.get(key, "")):   # keep the richest text per entity
                rep_text[key] = text

        for relation in result.relations:
            # Only link entities seen in this chunk, so both endpoints exist.
            if relation.source not in keys or relation.target not in keys:
                continue
            relation_records.append({
                "source_id": keys[relation.source], "target_id": keys[relation.target],
                "relation_type": relation.type, "description": relation.description,
                "document_id": document_id,
            })

    if not entity_records:
        return

    # Embed one vector per unique entity; reuse it for every mention record of that entity.
    if cfg.embed_entities:
        try:
            unique_keys = list(rep_text)
            vectors = await embedding_provider.embed_documents([rep_text[k] for k in unique_keys])
            vector_by_key = dict(zip(unique_keys, vectors))
            for record in entity_records:
                record["embedding"] = vector_by_key.get(record["id"])
        except Exception as exc:
            # Non-breaking: store entities without vectors — string matching still works.
            logger.warning("Entity embedding failed for %s; storing without vectors: %s", document_id, exc)
            logger.debug("Entity embedding failure (doc=%s)", document_id, exc_info=True)

    # Non-breaking: graph population must not fail ingestion — chunks are already indexed.
    try:
        await graph_store.upsert_entities(entity_records, user_id)
        await graph_store.upsert_relations(relation_records, user_id)
    except Exception as exc:
        logger.warning("Graph store write failed for %s: %s", document_id, exc)
        logger.debug("Graph store write failure (doc=%s)", document_id, exc_info=True)
        return

    # Community rebuild is decoupled from ingest by default (it re-clusters the whole tenant graph).
    # Auto-run only when explicitly opted in; otherwise call the rebuild endpoint on a schedule.
    community_cfg = settings.enrichment.community_detection
    auto_communities = community_cfg.enabled and community_cfg.rebuild_on_ingest
    if cfg.summarize_descriptions or auto_communities:
        from app.services.graph.service import GraphRAGService
        graph_service = GraphRAGService(graph_store, llm, embedding_provider)
        if cfg.summarize_descriptions:
            await graph_service.summarize_entity_descriptions(list(rep_text), user_id)
        if auto_communities:
            await graph_service.rebuild_communities(user_id)
