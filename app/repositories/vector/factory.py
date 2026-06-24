from app.repositories.base import BaseVectorDB
from app.utils.config import get_settings


def create_vector_db() -> BaseVectorDB:
    settings = get_settings()
    provider = settings.vector_store.provider

    if provider == "qdrant":
        from app.repositories.vector.qdrant import QdrantRepository
        cfg = settings.vector_store
        return QdrantRepository(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            collection_name=cfg.collection,
            sparse_model=cfg.sparse_model,
        )

    if provider == "pgvector":
        from app.repositories.vector.pgvector import PgvectorRepository
        from app.repositories.relational.database import get_engine
        return PgvectorRepository(engine=get_engine(), table=settings.vector_store.pgvector_table)

    if provider == "chroma":
        from app.repositories.vector.chroma import ChromaRepository
        cfg = settings.vector_store
        return ChromaRepository(path=cfg.chroma_path, collection_name=cfg.collection)

    raise ValueError(f"Unknown vector_store.provider: {provider!r}")
