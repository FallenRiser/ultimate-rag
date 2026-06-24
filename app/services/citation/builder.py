from typing import Any, Dict, List

from app.models.query import Citation, SourceChunk


def build_citation(chunks: List[Dict[str, Any]]) -> Citation:
    source_chunks = []
    scores = []

    for chunk in chunks:
        payload = chunk.get("payload", {})
        metadata = payload.get("metadata", {})
        score = float(chunk.get("rerank_score", chunk.get("score", 0.0)))
        scores.append(score)
        source_chunks.append(SourceChunk(
            chunk_id=str(chunk["id"]),
            document_id=metadata.get("document_id", ""),
            text=payload.get("text", ""),
            page=metadata.get("page"),
            score=score,
            retrieved_by=chunk.get("sources", []),
            metadata=metadata,
        ))

    confidence = sum(scores) / len(scores) if scores else 0.0
    return Citation(source_chunks=source_chunks, confidence=min(confidence, 1.0))
