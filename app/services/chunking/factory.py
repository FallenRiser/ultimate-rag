from app.services.chunking.base import BaseChunker
from app.utils.config import get_settings


def create_chunker() -> BaseChunker:
    settings = get_settings()
    cfg = settings.chunking
    strategy = cfg.strategy

    if strategy == "fixed":
        from app.services.chunking.fixed import FixedChunker
        return FixedChunker(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    if strategy in ("recursive", "document_aware"):
        from app.services.chunking.recursive import RecursiveChunker
        return RecursiveChunker(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    if strategy == "semantic":
        from app.services.chunking.semantic import SemanticChunker
        return SemanticChunker(threshold=cfg.semantic_threshold, max_tokens=cfg.chunk_size)

    raise ValueError(f"Unknown chunking.strategy: {strategy!r}")
