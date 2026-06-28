import hashlib
import json
import logging
from typing import Optional

from app.models.query import QueryRequest, QueryResponse
from app.observability.timing import StageTimer
from app.observability.tracing import log_chunks_to_trace, request_trace
from app.prompts.retrieval import ANSWER_SYSTEM
from app.repositories.base import BaseCacheBackend, BaseVectorDB
from app.services.citation.builder import build_citation
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.llm.base import BaseLLMProvider
from app.services.reranking.base import BaseReranker
from app.services.retrieval.service import RetrievalService
from app.utils.config import get_settings
from app.utils.text import cap_text

logger = logging.getLogger(__name__)


def _cache_key(user_id: str, query: str, mode: str, filters: dict) -> str:
    content = f"{user_id}:{query.lower().strip()}:{mode}:{json.dumps(filters, sort_keys=True)}"
    return hashlib.sha256(content.encode()).hexdigest()


class QueryPipeline:
    def __init__(
        self,
        vector_db: BaseVectorDB,
        embedding_provider: BaseEmbeddingProvider,
        reranker: Optional[BaseReranker],
        llm_provider: BaseLLMProvider,
        cache: Optional[BaseCacheBackend] = None,
        graph_store=None,
    ):
        self.retrieval = RetrievalService(vector_db, graph_store=graph_store, llm_provider=llm_provider)
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.llm_provider = llm_provider
        self.cache = cache

    async def run(self, request: QueryRequest) -> QueryResponse:
        settings = get_settings()
        mode = request.mode or settings.retrieval.default_mode
        top_k = request.top_k or settings.retrieval.top_k

        logger.info("Received query (mode=%s)", mode)
        logger.debug(
            "Received query: %r user_id=%s mode=%s top_k=%s filters=%s",
            request.query, request.user_id, mode, top_k, request.filters,
        )

        # One root span per request so the embedding + LLM calls nest under a single trace.
        with request_trace(
            "rag_query",
            {"query": request.query, "user_id": request.user_id, "mode": mode},
        ) as span:
            # Cache check
            if self.cache:
                key = _cache_key(request.user_id, request.query, mode, request.filters)
                cached = await self.cache.get(key)
                if cached:
                    logger.info("Cache hit — returning cached answer")
                    logger.debug("Cache hit for query: %s…", cap_text(request.query, settings.logging.preview_chars))
                    return QueryResponse.model_validate_json(cached)

            timer = StageTimer()

            # 1. Embed query
            logger.info("Embedding query")
            with timer.track("embedding"):
                query_vector = await self.embedding_provider.embed_query(request.query)

            # 1b. Agentic auto-filter: infer metadata filters from the query (config-gated).
            #     Explicit request.filters always win over inferred ones.
            auto = settings.agent.auto_filter
            inferred: dict = {}
            if auto.enabled:
                from app.repositories.relational.database import get_engine
                from app.services.retrieval.autofilter import infer_filters
                logger.info("Inferring metadata filters from query")
                with timer.track("auto_filter"):
                    inferred = await infer_filters(
                        self.llm_provider, request.query, request.user_id, get_engine(), auto
                    )
                if inferred:
                    logger.debug("Inferred filters: %s", inferred)
            effective_filters = {**inferred, **request.filters}

            # 2. Retrieve
            with timer.track("retrieval"):
                chunks = await self.retrieval.retrieve(
                    query=request.query,
                    query_vector=query_vector,
                    user_id=request.user_id,
                    mode=mode,
                    top_k=top_k,
                    filters=effective_filters,
                )
                # Soft fallback: if auto-inferred filters zeroed out results, retry without them.
                if not chunks and inferred:
                    logger.info("Auto-filter %s returned nothing; retrying without inferred filters", inferred)
                    chunks = await self.retrieval.retrieve(
                        query=request.query,
                        query_vector=query_vector,
                        user_id=request.user_id,
                        mode=mode,
                        top_k=top_k,
                        filters=request.filters,
                    )
            logger.info("Retrieved %d chunks (mode=%s)", len(chunks), mode)

            # 3. Rerank
            if self.reranker and chunks:
                logger.info("Reranking %d chunks", len(chunks))
                with timer.track("rerank"):
                    chunks = await self.reranker.rerank(request.query, chunks, settings.retrieval.rerank_top_k)

            log_chunks_to_trace(span, chunks)

            # 4. Citation
            citation = build_citation(chunks)

            # 5. Synthesize
            logger.info("Synthesizing answer from %d chunks", len(chunks))
            context_text = "\n\n---\n\n".join(
                c["payload"]["text"]
                for c in chunks
                if c.get("payload", {}).get("text")
            )
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {request.query}"},
            ]
            with timer.track("llm_synthesis"):
                answer = await self.llm_provider.chat(messages)

            response = QueryResponse(
                answer=answer,
                citations=citation,
                mode_used=mode,
                session_id=request.session_id,
                app_timings=timer.with_total(),
            )
            if span is not None:
                span.set_outputs({"answer": answer, "app_timings": response.app_timings})
            logger.info("Query complete: %s", response.app_timings)

            # Cache store
            if self.cache:
                await self.cache.set(key, response.model_dump_json(), settings.cache.ttl_seconds)

            return response


def get_query_pipeline() -> QueryPipeline:
    from app.repositories.cache.factory import create_cache
    from app.repositories.graph.factory import create_graph_store
    from app.repositories.vector.factory import create_vector_db
    from app.services.embeddings.factory import create_embedding_provider
    from app.services.llm.factory import create_llm_provider
    from app.services.reranking.factory import create_reranker

    settings = get_settings()
    return QueryPipeline(
        vector_db=create_vector_db(),
        embedding_provider=create_embedding_provider(),
        reranker=create_reranker() if settings.reranker.enabled else None,
        llm_provider=create_llm_provider(),
        cache=create_cache() if settings.cache.enabled else None,
        graph_store=create_graph_store() if settings.graph_store.enabled else None,
    )
