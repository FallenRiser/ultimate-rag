import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from app.repositories.base import BaseGraphStore, BaseVectorDB

logger = logging.getLogger(__name__)

_VALID_MODES = {"semantic", "bm25", "graph", "hybrid", "hybrid_graph"}


class RetrievalService:
    def __init__(self, vector_db: BaseVectorDB, graph_store: Optional[BaseGraphStore] = None):
        self.vector_db = vector_db
        self.graph_store = graph_store

    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        user_id: str,
        mode: str,
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown retrieval mode: {mode!r}. Valid: {_VALID_MODES}")

        if mode in ("bm25", "hybrid", "hybrid_graph") and not self.vector_db.supports_sparse:
            raise ValueError(
                f"Mode {mode!r} requires BM25/sparse support but the active store "
                f"({self.vector_db.__class__.__name__}) is semantic-only. "
                "Set vector_store.provider='qdrant' or use mode='semantic'."
            )

        if mode in ("graph", "hybrid_graph") and not self.graph_store:
            raise ValueError(
                f"Mode {mode!r} requires a graph store but none is configured. "
                "Set graph_store.enabled=true in config."
            )

        logger.debug("Retrieving: mode=%s top_k=%d filters=%s", mode, top_k, filters)

        if mode == "semantic":
            return _tag(await self.vector_db.dense_search(query_vector, user_id, top_k, filters), "dense")

        if mode == "bm25":
            return _tag(await self.vector_db.sparse_search(query, user_id, top_k, filters), "bm25")

        if mode == "hybrid":
            dense, sparse = await asyncio.gather(
                self.vector_db.dense_search(query_vector, user_id, top_k, filters),
                self.vector_db.sparse_search(query, user_id, top_k, filters),
            )
            return _rrf_fuse(_tag(dense, "dense"), _tag(sparse, "bm25"), top_k)

        if mode == "graph":
            return _tag(await self._graph_retrieve(query, user_id, top_k), "graph")

        if mode == "hybrid_graph":
            dense, sparse, graph = await asyncio.gather(
                self.vector_db.dense_search(query_vector, user_id, top_k, filters),
                self.vector_db.sparse_search(query, user_id, top_k, filters),
                self._graph_retrieve(query, user_id, top_k),
            )
            fused = _rrf_fuse(_tag(dense, "dense"), _tag(sparse, "bm25"), top_k)
            return _rrf_fuse(fused, _tag(graph, "graph"), top_k)

        raise ValueError(f"Unhandled mode: {mode!r}")

    async def _graph_retrieve(self, query: str, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Extract entity names from query, find graph nodes that mention them,
        then hydrate their chunk_ids back into real chunks from the vector store."""
        entity_names = _extract_entity_names(query)
        chunk_ids = await self.graph_store.find_chunks_for_entities(entity_names, user_id)
        logger.debug("Graph: entities=%s -> %d chunk_ids", entity_names, len(chunk_ids))
        if not chunk_ids:
            return []
        return await self.vector_db.fetch_by_ids(chunk_ids[:top_k], user_id)


_STOPWORDS = {
    "what", "are", "is", "the", "a", "an", "of", "to", "for", "and", "or", "in", "on",
    "at", "about", "me", "my", "tell", "give", "show", "list", "how", "why", "when",
    "where", "who", "which", "was", "were", "do", "does", "did", "with", "this", "that",
    "these", "those", "please", "can", "could", "you", "your", "it", "its", "their",
    "there", "from", "by", "as", "be", "have", "has", "had", "results", "result",
}


def _extract_entity_names(text: str) -> List[str]:
    """Candidate terms to match against graph entity names. Lowercased so matching is
    case-insensitive: quoted phrases, capitalised phrases, and content words (>=3 chars,
    minus stopwords). Matched with partial CONTAINS in the graph store."""
    quoted = re.findall(r'"([^"]+)"', text)
    capitalised = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) if w.lower() not in _STOPWORDS]
    terms = {t.lower() for t in (*quoted, *capitalised, *words)}
    return list(terms)[:12]


def _tag(results: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Record which retriever produced each result, for per-chunk provenance."""
    for result in results:
        result["sources"] = [source]
    return results


def _rrf_fuse(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    top_k: int,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion — combine two ranked lists, unioning their source tags."""
    rrf_scores: Dict[str, float] = {}
    all_results: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, List[str]] = {}

    for results in (left, right):
        for rank, result in enumerate(results):
            chunk_id = str(result["id"])
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            all_results.setdefault(chunk_id, result)
            for source in result.get("sources", []):
                sources.setdefault(chunk_id, [])
                if source not in sources[chunk_id]:
                    sources[chunk_id].append(source)

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]
    return [
        {**all_results[cid], "score": rrf_scores[cid], "sources": sources.get(cid, [])}
        for cid in sorted_ids
    ]
