import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from app.models.query import QueryEntities
from app.prompts.retrieval import QUERY_ENTITY_SYSTEM
from app.repositories.base import BaseGraphStore, BaseVectorDB
from app.services.llm.base import BaseLLMProvider
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

_VALID_MODES = {"semantic", "bm25", "graph", "hybrid", "hybrid_graph", "graph_global"}


class RetrievalService:
    def __init__(
        self,
        vector_db: BaseVectorDB,
        graph_store: Optional[BaseGraphStore] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        self.vector_db = vector_db
        self.graph_store = graph_store
        self.llm_provider = llm_provider   # only needed for retrieval.graph_query_entities="llm"

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

        if mode in ("graph", "hybrid_graph", "graph_global") and not self.graph_store:
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
            # _graph_retrieve already tags its own results (graph chunks + community summaries).
            return await self._graph_retrieve(query, query_vector, user_id, top_k)

        if mode == "graph_global":
            return await self._global_retrieve(query, query_vector, user_id, top_k)

        if mode == "hybrid_graph":
            dense, sparse, graph = await asyncio.gather(
                self.vector_db.dense_search(query_vector, user_id, top_k, filters),
                self.vector_db.sparse_search(query, user_id, top_k, filters),
                self._graph_retrieve(query, query_vector, user_id, top_k),
            )
            fused = _rrf_fuse(_tag(dense, "dense"), _tag(sparse, "bm25"), top_k)
            return _rrf_fuse(fused, graph, top_k)

        raise ValueError(f"Unhandled mode: {mode!r}")

    async def _graph_retrieve(
        self, query: str, query_vector: List[float], user_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Local GraphRAG: pick seed entities (semantic, falling back to string match), expand
        their multi-hop neighbourhood into chunks, and — dual-level — blend in the community
        summaries of those seeds. Returns chunks already tagged by source."""
        rcfg = get_settings().retrieval

        seeds = await self.graph_store.search_entities(query_vector, user_id, rcfg.graph_seed_top_k)
        if not seeds:
            names = await self._query_entities(query)
            seeds = await self.graph_store.match_entities_by_name(names, user_id)
        seed_ids = [s["id"] for s in seeds]
        logger.debug("Graph: %d seed entities, hops=%d", len(seed_ids), rcfg.graph_hops)

        chunks: List[Dict[str, Any]] = []
        if seed_ids:
            chunk_ids = await self.graph_store.expand_and_collect(seed_ids, user_id, rcfg.graph_hops)
            if chunk_ids:
                ranked = chunk_ids[:top_k]
                fetched = await self.vector_db.fetch_by_ids(ranked, user_id)
                # fetch_by_ids does NOT preserve input order — restore the graph ranking (seeds
                # first, then by hop distance) so it survives when no reranker runs.
                position = {cid: i for i, cid in enumerate(ranked)}
                fetched.sort(key=lambda c: position.get(str(c["id"]), len(position)))
                chunks = _tag(fetched, "graph")

        # Dual-level: prepend the seeds' community reports as high-level context.
        if rcfg.graph_dual_level and seed_ids:
            communities = await self.graph_store.communities_for_entities(seed_ids, user_id)
            if communities:
                logger.debug("Graph dual-level: %d community summaries", len(communities))
                chunks = _community_chunks(communities) + chunks
        return chunks

    async def _global_retrieve(
        self, query: str, query_vector: List[float], user_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Global GraphRAG: answer thematic questions from the community reports, ranked by
        similarity to the query (no entity anchoring). If communities were never built (detection
        off), degrade to local graph retrieval, then to plain semantic — never a silent empty answer."""
        communities = await self.graph_store.search_communities(query_vector, user_id, top_k)
        if communities:
            logger.debug("Graph global: %d community summaries", len(communities))
            return _community_chunks(communities)

        logger.info("graph_global: no community reports; falling back to local graph then semantic")
        local = await self._graph_retrieve(query, query_vector, user_id, top_k)
        if local:
            return local
        return _tag(await self.vector_db.dense_search(query_vector, user_id, top_k, {}), "dense")

    async def _query_entities(self, query: str) -> List[str]:
        """Seed entity names for graph retrieval. Default 'regex' is free; 'llm' is one extra
        structured call with better recall. LLM failures or empty results fall back to regex."""
        cfg = get_settings().retrieval
        if cfg.graph_query_entities != "llm" or self.llm_provider is None:
            return _extract_entity_names(query)
        try:
            result: QueryEntities = await self.llm_provider.structured_output(
                messages=[
                    {"role": "system", "content": QUERY_ENTITY_SYSTEM},
                    {"role": "user", "content": query},
                ],
                schema=QueryEntities,
            )
        except Exception as exc:
            # Non-breaking: fall back to the regex extractor so graph retrieval still runs.
            logger.warning("LLM query-entity extraction failed; using regex: %s", exc)
            logger.debug("LLM query-entity extraction failure", exc_info=True)
            return _extract_entity_names(query)

        names = [n.strip() for n in result.entities if n.strip()]
        if names:
            logger.debug("LLM query entities: %s", names)
            return names[:12]
        logger.debug("LLM found no query entities; falling back to regex")
        return _extract_entity_names(query)


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


def _community_chunks(communities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Wrap community reports as chunk-shaped results so they flow through synthesis + citations
    like any retrieved chunk (tagged 'community', no source document)."""
    chunks = []
    for community in communities:
        summary = community.get("summary")
        if not summary:
            continue
        chunks.append({
            "id": f"community::{community['id']}",
            "score": float(community.get("score", 0.0)),
            "payload": {
                "text": summary,
                "metadata": {
                    "document_id": "",
                    "community_id": community["id"],
                    "title": community.get("title", ""),
                },
            },
            "sources": ["community"],
        })
    return chunks


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


if __name__ == "__main__":
    # Self-check for _query_entities: llm vs regex selection and both fallbacks.
    import asyncio

    class _FakeLLM(BaseLLMProvider):
        def __init__(self, entities: List[str], fail: bool = False):
            self._entities, self._fail = entities, fail

        async def chat(self, messages, **kwargs) -> str:
            return ""

        async def structured_output(self, messages, schema, **kwargs):
            if self._fail:
                raise RuntimeError("llm down")
            return QueryEntities(entities=self._entities)

    async def _demo() -> None:
        settings = get_settings()
        settings.retrieval.graph_query_entities = "llm"

        svc = RetrievalService(vector_db=None, llm_provider=_FakeLLM(["Nvidia", "  ", "CUDA"]))
        assert await svc._query_entities("about nvidia") == ["Nvidia", "CUDA"]      # LLM, blanks trimmed

        failing = RetrievalService(vector_db=None, llm_provider=_FakeLLM([], fail=True))
        assert await failing._query_entities("about Nvidia") != []                  # raise -> regex fallback

        empty = RetrievalService(vector_db=None, llm_provider=_FakeLLM([]))
        assert "nvidia" in await empty._query_entities("about Nvidia")              # empty -> regex fallback

        settings.retrieval.graph_query_entities = "regex"
        regex = RetrievalService(vector_db=None, llm_provider=_FakeLLM(["UNUSED"]))
        assert "UNUSED" not in await regex._query_entities("about Nvidia")          # regex mode ignores LLM

        # graph_global fallback: no communities -> local graph (also empty here) -> semantic.
        class _EmptyGraph:
            async def search_communities(self, qv, uid, k): return []
            async def search_entities(self, qv, uid, k): return []
            async def match_entities_by_name(self, names, uid): return []
            async def communities_for_entities(self, ids, uid): return []
            async def expand_and_collect(self, ids, uid, hops): return []

        class _FakeVec:
            async def dense_search(self, qv, uid, k, filters):
                return [{"id": "x", "score": 0.5, "payload": {"text": "t"}}]

        glob = RetrievalService(vector_db=_FakeVec(), graph_store=_EmptyGraph(), llm_provider=_FakeLLM([]))
        out = await glob._global_retrieve("main themes?", [1.0, 0.0], "u", 5)
        assert out and out[0]["sources"] == ["dense"]                               # degraded to semantic, not empty

        # Graph ranking must survive hydration even though fetch_by_ids reorders.
        class _RankGraph:
            async def search_entities(self, qv, uid, k): return [{"id": "e1", "name": "E", "score": 1.0}]
            async def match_entities_by_name(self, names, uid): return []
            async def expand_and_collect(self, ids, uid, hops): return ["c1", "c2", "c3"]
            async def communities_for_entities(self, ids, uid): return []

        class _ShuffleVec:
            async def fetch_by_ids(self, ids, uid):
                return [{"id": cid, "score": 1.0, "payload": {"text": cid}} for cid in ("c3", "c1", "c2")]

        gr = RetrievalService(vector_db=_ShuffleVec(), graph_store=_RankGraph(), llm_provider=_FakeLLM([]))
        ranked = await gr._graph_retrieve("q", [1.0, 0.0], "u", 10)
        assert [c["id"] for c in ranked] == ["c1", "c2", "c3"], "graph ranking lost at hydration"
        print("OK")

    asyncio.run(_demo())
