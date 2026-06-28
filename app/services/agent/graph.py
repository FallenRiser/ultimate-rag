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

from app.models.agent import AgentState, QueryAnalysis, RelevanceGrade, StandaloneQuery
from app.observability.tracing import log_chunks_to_trace, request_trace
from app.prompts.agent import ANALYZE_SYSTEM, CONTEXTUALIZE_SYSTEM, GRADE_SYSTEM, SYNTHESIZE_SYSTEM
from app.repositories.base import BaseVectorDB
from app.services.citation.builder import build_citation
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.llm.base import BaseLLMProvider
from app.services.reranking.base import BaseReranker
from app.services.retrieval.service import _VALID_MODES, RetrievalService, _rrf_fuse
from app.utils.config import get_settings
from app.utils.text import cap_list

logger = logging.getLogger(__name__)


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
        llm = create_llm_provider()
        _services = _AgentServices(
            retrieval=RetrievalService(create_vector_db(), graph_store=create_graph_store(), llm_provider=llm),
            embedding=create_embedding_provider(),
            reranker=create_reranker() if settings.reranker.enabled else None,
            llm=llm,
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
            # Accumulate so a looped node's time is summed across iterations, not overwritten.
            prior = state.app_timings.get(stage, 0.0)
            return {**update, "app_timings": {**state.app_timings, stage: round(prior + elapsed_ms, 1)}}
        return wrapper
    return decorator

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
                {"role": "system", "content": CONTEXTUALIZE_SYSTEM},
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

    snippets = [c.get("payload", {}).get("text", "") for c in chunks]
    context = "\n---\n".join(s for s in snippets if s)
    logger.debug("Exploration retrieved %d snippets for grounding", len(snippets))
    return {"context": context}


@_node("query_analysis")
async def analyze_query(state: AgentState) -> dict:
    """LLM decides if the query needs rewriting/decomposition and which retrieval mode, and
    (optionally) infers metadata filters. Filter inference and the analysis call are
    independent, so they run concurrently. Explicit (request) filters win over inferred ones."""
    settings = get_settings()
    svc = _get_services()
    auto = settings.agent.auto_filter
    do_analysis = settings.agent.query_rewrite or settings.agent.decompose

    explicit_filters = dict(state.filters)   # request filters, before inferred ones are merged

    if state.context:
        user_content = f"Context from the knowledge base:\n{state.context}\n\nUser query: {state.query}"
    else:
        user_content = state.query

    async def _infer() -> dict:
        if not auto.enabled:
            return {}
        from app.repositories.relational.database import get_engine
        from app.services.retrieval.autofilter import infer_filters
        return await infer_filters(svc.llm, state.query, state.user_id, get_engine(), auto)

    async def _analyze() -> Optional[QueryAnalysis]:
        if not do_analysis:
            return None
        try:
            return await svc.llm.structured_output(
                messages=[
                    {"role": "system", "content": ANALYZE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                schema=QueryAnalysis,
            )
        except Exception as exc:
            logger.warning("Query analysis LLM call failed; using defaults: %s", exc)
            logger.debug("Query analysis failure (query=%r)", state.query, exc_info=True)
            return None

    inferred, analysis = await asyncio.gather(_infer(), _analyze())

    filter_update = {"filters": {**inferred, **state.filters}} if inferred else {}

    if analysis is None:
        mode = state.mode or settings.retrieval.default_mode
        return {"mode": mode, "subqueries": [state.query], "explicit_filters": explicit_filters, **filter_update}

    # Clamp the LLM's suggested mode to a real mode; the retrieve node still guards against a
    # valid-but-store-incompatible mode (e.g. hybrid on a semantic-only store).
    suggested = analysis.suggested_mode if analysis.suggested_mode in _VALID_MODES else None
    mode = state.mode or suggested or settings.retrieval.default_mode
    rewritten = analysis.rewritten_query or state.query
    subqueries = analysis.subqueries if (analysis.needs_decomposition and analysis.subqueries) else [rewritten]
    return {
        "query": rewritten,
        "mode": mode,
        "subqueries": subqueries[: settings.agent.max_subqueries],
        "explicit_filters": explicit_filters,
        **filter_update,
    }


@_node("retrieval")
async def retrieve(state: AgentState) -> dict:
    """Retrieves chunks for each subquery in parallel, then RRF-fuses the lists. Guards against an
    unavailable mode, and broadens the search on a corrective (graded-irrelevant) retry."""
    svc = _get_services()
    settings = get_settings()
    top_k = settings.retrieval.top_k
    is_retry = state.loop_count > 0

    if is_retry:
        # The grade failed last time; re-running the same query reproduces the same chunks. Broaden:
        # drop inferred filters (keep explicit), widen top_k, and use semantic (always available).
        mode = "semantic"
        filters = state.explicit_filters
        top_k *= 2
        logger.info("Corrective retry %d: broadening to semantic, top_k=%d, explicit filters=%s",
                    state.loop_count, top_k, filters)
    else:
        mode = state.mode or settings.retrieval.default_mode
        filters = state.filters

    async def _retrieve_one(subquery: str, mode: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_vector = await svc.embedding.embed_query(subquery)
        return await svc.retrieval.retrieve(
            query=subquery, query_vector=query_vector, user_id=state.user_id,
            mode=mode, top_k=top_k, filters=filters,
        )

    async def _retrieve_all(mode: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        subqueries = state.subqueries or [state.query]
        all_results = await asyncio.gather(*[_retrieve_one(q, mode, filters) for q in subqueries])
        if len(all_results) == 1:
            return all_results[0]
        fused = all_results[0]
        for result_list in all_results[1:]:
            fused = _rrf_fuse(fused, result_list, top_k)
        return fused

    # Mode safety: the chosen mode may be LLM-suggested or incompatible with the active store
    # (e.g. hybrid on a semantic-only store, a graph mode with no graph store). Degrade to
    # semantic instead of rejecting a valid query.
    try:
        chunks = await _retrieve_all(mode, filters)
    except (ValueError, NotImplementedError) as exc:
        logger.warning("Retrieval mode %r unavailable (%s); falling back to semantic", mode, exc)
        logger.debug("Mode fallback (query=%r)", state.query, exc_info=True)
        mode = "semantic"
        chunks = await _retrieve_all(mode, filters)

    # Soft fallback (first pass only — the retry path already uses explicit filters): if filters
    # zero out results, drop the INFERRED filters and retry with the user's explicit filters.
    if not chunks and not is_retry and state.filters and state.explicit_filters != state.filters:
        logger.info("Filters %s returned nothing; retrying with explicit filters %s",
                    state.filters, state.explicit_filters)
        chunks = await _retrieve_all(mode, state.explicit_filters)

    # Increment so _should_loop actually bounds the grade→retrieve loop by max_retrieval_loops.
    return {"retrieved_chunks": chunks, "loop_count": state.loop_count + 1}


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

    graded = cap_list(chunks, settings.agent.grade_max_chunks)
    context = "\n\n".join(
        c.get("payload", {}).get("text", "")
        for c in graded
    )
    try:
        grade: RelevanceGrade = await svc.llm.structured_output(
            messages=[
                {"role": "system", "content": GRADE_SYSTEM},
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

    # Nothing retrieved → don't spend an LLM call asking it to answer from empty context.
    if not context.strip():
        logger.info("No context retrieved; skipping synthesis")
        return {"answer": "I couldn't find anything relevant in the knowledge base to answer that."}

    messages = [
        {"role": "system", "content": SYNTHESIZE_SYSTEM},
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

def _build_graph():
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

    return builder.compile()


_compiled_graph = None


def _get_graph():
    """Compile the agent graph once and reuse it — it's stateless (state is passed at invoke),
    so rebuilding it per request is pure waste."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


async def run_agent(state: AgentState) -> AgentState:
    # Conversation memory comes from state.history (loaded from chat_messages and consumed by
    # the contextualize node) — no LangGraph checkpointer, since ainvoke passes the full state
    # each turn and the checkpoint would just be overwritten.
    graph = _get_graph()

    # One root span per request so every node's embedding/LLM call nests in one trace.
    with request_trace(
        "rag_agent",
        {"query": state.query, "user_id": state.user_id, "mode": state.mode},
    ) as span:
        result = await graph.ainvoke(state.model_dump())
        final = AgentState(**result)
        log_chunks_to_trace(span, final.reranked_chunks or final.retrieved_chunks)
        if span is not None:
            # The mode is decided inside analyze_query, so record the resolved one on the output.
            span.set_outputs({"answer": final.answer, "mode": final.mode, "app_timings": final.app_timings})
        return final


if __name__ == "__main__":
    # Self-check: retrieve-node mode safety (#1) and corrective broadening (#2).
    class _FakeEmb:
        async def embed_query(self, text):
            return [0.0]

    class _FakeRetrieval:
        def __init__(self):
            self.calls: List[Dict[str, Any]] = []

        async def retrieve(self, query, query_vector, user_id, mode, top_k, filters):
            self.calls.append({"mode": mode, "top_k": top_k, "filters": dict(filters)})
            if mode != "semantic":
                raise ValueError(f"mode {mode!r} unavailable")   # only semantic "works" here
            return [{"id": "c1", "score": 1.0, "payload": {"text": "t"}}]

    async def _demo() -> None:
        global _services
        fake = _FakeRetrieval()
        _services = _AgentServices(retrieval=fake, embedding=_FakeEmb(), reranker=None, llm=None)
        base_top_k = get_settings().retrieval.top_k

        # #1: an unavailable mode degrades to semantic instead of raising.
        out = await retrieve(AgentState(query="q", user_id="u", mode="hybrid", subqueries=["q"]))
        assert out["retrieved_chunks"], "should fall back to semantic and return chunks"
        assert fake.calls[-1]["mode"] == "semantic"

        # #2: a corrective retry broadens — semantic, doubled top_k, explicit filters only.
        fake.calls.clear()
        out2 = await retrieve(AgentState(
            query="q", user_id="u", mode="graph", subqueries=["q"], loop_count=1,
            filters={"a": "1", "b": "2"}, explicit_filters={"a": "1"},
        ))
        last = fake.calls[-1]
        assert last["mode"] == "semantic"
        assert last["filters"] == {"a": "1"}            # inferred filter "b" dropped
        assert last["top_k"] == base_top_k * 2          # widened
        assert out2["loop_count"] == 2
        print("OK")

    asyncio.run(_demo())
