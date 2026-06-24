"""
LangGraph agent for multi-step RAG.

Node sequence:
  analyze_query → rewrite → decompose → route →
  retrieve (parallel per subquery) → fuse (RRF) →
  rerank → grade → synthesize → cite

Grade can loop back to retrieve (up to agent.max_retrieval_loops).
"""

import asyncio
import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from app.repositories.base import BaseVectorDB
from app.services.citation.builder import build_citation
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.llm.base import BaseLLMProvider
from app.services.reranking.base import BaseReranker
from app.services.retrieval.service import RetrievalService, _rrf_fuse
from app.utils.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None
    mode: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)  # prior {role, content} turns
    context: str = ""                                            # snippets from exploratory retrieval
    app_timings: Dict[str, float] = Field(default_factory=dict)  # per-node milliseconds
    subqueries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    loop_count: int = 0
    grade_passed: bool = False


# ---------------------------------------------------------------------------
# Structured output schemas (used by LLM nodes)
# ---------------------------------------------------------------------------

class QueryAnalysis(BaseModel):
    needs_decomposition: bool = False
    subqueries: List[str] = Field(default_factory=list)
    suggested_mode: str = "hybrid"
    rewritten_query: Optional[str] = None


class RelevanceGrade(BaseModel):
    is_relevant: bool = True
    reason: str = ""


class StandaloneQuery(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Shared services (lazy module-level singleton)
# ---------------------------------------------------------------------------

@dataclass
class _AgentServices:
    retrieval: RetrievalService
    embedding: BaseEmbeddingProvider
    reranker: Optional[BaseReranker]
    llm: BaseLLMProvider


_services: Optional[_AgentServices] = None


def _get_services() -> _AgentServices:
    global _services
    if _services is None:
        from app.repositories.graph.factory import create_graph_store
        from app.repositories.vector.factory import create_vector_db
        from app.services.embeddings.factory import create_embedding_provider
        from app.services.llm.factory import create_llm_provider
        from app.services.reranking.factory import create_reranker

        settings = get_settings()
        _services = _AgentServices(
            retrieval=RetrievalService(create_vector_db(), graph_store=create_graph_store()),
            embedding=create_embedding_provider(),
            reranker=create_reranker() if settings.reranker.enabled else None,
            llm=create_llm_provider(),
        )
    return _services


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _node(stage: str) -> Callable:
    """Wrap an agent node so it logs its stage and records its duration into app_timings."""
    def decorator(fn: Callable[["AgentState"], Awaitable[dict]]) -> Callable:
        @functools.wraps(fn)
        async def wrapper(state: "AgentState") -> dict:
            logger.info("Agent stage: %s", stage)
            start = time.perf_counter()
            update = await fn(state)
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)
            return {**update, "app_timings": {**state.app_timings, stage: elapsed_ms}}
        return wrapper
    return decorator

_CONTEXTUALIZE_SYSTEM = (
    "You rewrite a user's new question into a self-contained question. "
    "Resolve references such as 'it', 'that', 'they', 'this', or an omitted subject using the "
    "conversation — but ONLY when the new question clearly depends on earlier turns. "
    "If the new question is already self-contained, or introduces a new topic or document, "
    "return it unchanged. Never add facts, names, or constraints the user did not state."
)


def _format_history(history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for turn in history:
        speaker = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.get('content', '')}")
    return "\n".join(lines)


@_node("contextualize")
async def contextualize(state: AgentState) -> dict:
    """Rewrite a follow-up into a standalone query using recent history (conservative).
    Retrieval and filters stay per-turn, so a topic switch is never poisoned by old context."""
    settings = get_settings()
    if not settings.chat.memory_enabled:
        return {}
    if not state.history:
        return {}

    svc = _get_services()
    user_prompt = (
        f"Conversation so far:\n{_format_history(state.history)}\n\n"
        f"New question: {state.query}\n\n"
        "Rewrite the new question to stand on its own only if it refers back to the "
        "conversation. Otherwise return it exactly as written."
    )
    try:
        result: StandaloneQuery = await svc.llm.structured_output(
            messages=[
                {"role": "system", "content": _CONTEXTUALIZE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            schema=StandaloneQuery,
        )
    except Exception as exc:
        logger.warning("Contextualization failed: %s", exc)
        return {}

    standalone = result.query.strip()
    if not standalone:
        return {}
    if standalone != state.query:
        logger.info("Contextualized query: %r -> %r", state.query, standalone)
    return {"query": standalone}


@_node("explore")
async def explore_context(state: AgentState) -> dict:
    """Lightweight semantic retrieval on the raw query, so analyze_query can rewrite/decompose
    with knowledge of what the documents actually contain (fills in implied subjects)."""
    settings = get_settings()
    if not settings.agent.context_exploration:
        return {}

    svc = _get_services()
    top_k = settings.agent.context_exploration_top_k
    query_vector = await svc.embedding.embed_query(state.query)
    # Always semantic here — independent of the requested mode (which may need graph/sparse).
    chunks = await svc.retrieval.vector_db.dense_search(query_vector, state.user_id, top_k)

    snippets = [c.get("payload", {}).get("text", "")[:300] for c in chunks]
    context = "\n---\n".join(s for s in snippets if s)
    logger.debug("Exploration retrieved %d snippets for grounding", len(snippets))
    return {"context": context}


_ANALYZE_SYSTEM = (
    "Analyze the user query. Determine if it needs decomposition into simpler subqueries. "
    "If it can be answered directly, set needs_decomposition=false and leave subqueries empty. "
    "Suggest a retrieval mode: 'semantic' for factual lookups, 'hybrid' for exploratory queries. "
    "If context from the knowledge base is provided, use it to ground your rewrite and to fill in "
    "implied subjects the user left out (e.g. which company or metric). Never invent facts."
)


@_node("query_analysis")
async def analyze_query(state: AgentState) -> dict:
    """LLM decides if query needs rewriting, decomposition, which retrieval mode, and
    (optionally) infers metadata filters from the query."""
    settings = get_settings()
    svc = _get_services()

    # Agentic auto-filter (independent of rewrite/decompose). Explicit filters win.
    filter_update = {}
    auto = settings.agent.auto_filter
    if auto.enabled:
        from app.services.retrieval.autofilter import infer_filters
        inferred = await infer_filters(svc.llm, state.query, auto.fields)
        if inferred:
            filter_update = {"filters": {**inferred, **state.filters}}

    if not settings.agent.query_rewrite and not settings.agent.decompose:
        mode = state.mode or settings.retrieval.default_mode
        return {"mode": mode, "subqueries": [state.query], **filter_update}

    if state.context:
        user_content = f"Context from the knowledge base:\n{state.context}\n\nUser query: {state.query}"
    else:
        user_content = state.query

    try:
        analysis: QueryAnalysis = await svc.llm.structured_output(
            messages=[
                {"role": "system", "content": _ANALYZE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            schema=QueryAnalysis,
        )
    except Exception as exc:
        logger.warning("Query analysis LLM call failed: %s", exc)
        return {"mode": state.mode or settings.retrieval.default_mode, "subqueries": [state.query]}

    mode = state.mode or analysis.suggested_mode or settings.retrieval.default_mode
    rewritten = analysis.rewritten_query or state.query
    subqueries = analysis.subqueries if (analysis.needs_decomposition and analysis.subqueries) else [rewritten]

    return {
        "query": rewritten,
        "mode": mode,
        "subqueries": subqueries[: settings.agent.max_subqueries],
        **filter_update,
    }


@_node("retrieval")
async def retrieve(state: AgentState) -> dict:
    """Retrieves chunks for each subquery in parallel, then RRF-fuses the lists."""
    svc = _get_services()
    settings = get_settings()
    mode = state.mode or settings.retrieval.default_mode
    top_k = settings.retrieval.top_k

    async def _retrieve_one(subquery: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_vector = await svc.embedding.embed_query(subquery)
        return await svc.retrieval.retrieve(
            query=subquery,
            query_vector=query_vector,
            user_id=state.user_id,
            mode=mode,
            top_k=top_k,
            filters=filters,
        )

    async def _retrieve_all(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        subqueries = state.subqueries or [state.query]
        all_results = await asyncio.gather(*[_retrieve_one(q, filters) for q in subqueries])
        if len(all_results) == 1:
            return all_results[0]
        fused = all_results[0]
        for result_list in all_results[1:]:
            fused = _rrf_fuse(fused, result_list, top_k)
        return fused

    chunks = await _retrieve_all(state.filters)

    # Soft fallback: never let filters zero out the result set.
    if not chunks and state.filters:
        logger.info("Filters %s returned nothing; retrying without filters", state.filters)
        chunks = await _retrieve_all({})

    return {"retrieved_chunks": chunks}


@_node("rerank")
async def rerank_chunks(state: AgentState) -> dict:
    """Reranks retrieved chunks using the BGE reranker (if available)."""
    svc = _get_services()
    settings = get_settings()

    if not svc.reranker or not state.retrieved_chunks:
        return {"reranked_chunks": state.retrieved_chunks}

    reranked = await svc.reranker.rerank(
        query=state.query,
        documents=list(state.retrieved_chunks),
        top_n=settings.retrieval.rerank_top_k,
    )
    return {"reranked_chunks": reranked}


@_node("grade")
async def grade_relevance(state: AgentState) -> dict:
    """LLM grades whether retrieved chunks are relevant to the query."""
    settings = get_settings()
    svc = _get_services()

    chunks = state.reranked_chunks or state.retrieved_chunks
    if not chunks:
        return {"grade_passed": False}

    if not settings.agent.grade_relevance:
        return {"grade_passed": True}

    context = "\n\n".join(
        c.get("payload", {}).get("text", "")[:300]
        for c in chunks[:3]
    )
    try:
        grade: RelevanceGrade = await svc.llm.structured_output(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are grading whether the retrieved context is relevant to the query. "
                        "Set is_relevant=true if the context contains information that helps answer the query."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {state.query}\n\nContext:\n{context}",
                },
            ],
            schema=RelevanceGrade,
        )
        return {"grade_passed": grade.is_relevant}
    except Exception as exc:
        logger.warning("Grade relevance LLM call failed: %s", exc)
        return {"grade_passed": True}  # proceed on error


@_node("llm_synthesis")
async def synthesize(state: AgentState) -> dict:
    """LLM generates an answer from the reranked chunks."""
    svc = _get_services()

    chunks = state.reranked_chunks or state.retrieved_chunks
    context = "\n\n---\n\n".join(
        c.get("payload", {}).get("text", "")
        for c in chunks
        if c.get("payload", {}).get("text")
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise, factual assistant. "
                "Answer the question using ONLY the provided context. "
                "If the context is insufficient, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {state.query}",
        },
    ]
    answer = await svc.llm.chat(messages)
    return {"answer": answer}


@_node("cite")
async def cite(state: AgentState) -> dict:
    """Builds citations from the reranked chunks."""
    chunks = state.reranked_chunks or state.retrieved_chunks
    citation = build_citation(chunks)
    return {"citations": [s.model_dump() for s in citation.source_chunks]}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def _should_loop(state: AgentState) -> str:
    settings = get_settings()
    if not state.grade_passed and state.loop_count < settings.agent.max_retrieval_loops:
        return "retrieve"   # loop back
    return "synthesize"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph(checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(AgentState)
    builder.add_node("contextualize", contextualize)
    builder.add_node("explore", explore_context)
    builder.add_node("analyze_query", analyze_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank_chunks)
    builder.add_node("grade", grade_relevance)
    builder.add_node("synthesize", synthesize)
    builder.add_node("cite", cite)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "explore")
    builder.add_edge("explore", "analyze_query")
    builder.add_edge("analyze_query", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "grade")
    builder.add_conditional_edges("grade", _should_loop, {"retrieve": "retrieve", "synthesize": "synthesize"})
    builder.add_edge("synthesize", "cite")
    builder.add_edge("cite", END)

    return builder.compile(checkpointer=checkpointer)


async def _get_checkpointer(session_id: Optional[str]):
    settings = get_settings()
    if not session_id:
        return None, None

    if settings.chat.checkpointer == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver(), None

    if settings.chat.checkpointer == "sqlite":
        # Persistent, zero-infra chat memory (requires langgraph-checkpoint-sqlite).
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            conn = await aiosqlite.connect(settings.chat.sqlite_path)
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            return saver, conn
        except Exception as exc:
            logger.warning("SQLite checkpointer unavailable, falling back to memory: %s", exc)
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver(), None

    # Postgres checkpointer (requires psycopg[binary]>=3.0)
    try:
        import psycopg
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        dsn = settings.database.dsn.replace("postgresql+asyncpg://", "postgresql://")
        conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        return saver, conn
    except Exception as exc:
        logger.warning("Postgres checkpointer unavailable, falling back to memory: %s", exc)
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver(), None


async def run_agent(state: AgentState) -> AgentState:
    checkpointer, conn = await _get_checkpointer(state.session_id)
    graph = _build_graph(checkpointer)

    config = (
        {"configurable": {"thread_id": state.session_id}}
        if state.session_id
        else {}
    )

    # Increment loop_count on every re-entry so the grade condition works
    state = state.model_copy(update={"loop_count": state.loop_count})

    try:
        result = await graph.ainvoke(state.model_dump(), config=config)
        return AgentState(**result)
    finally:
        if conn is not None:
            await conn.close()
