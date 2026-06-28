import logging
import time

from fastapi import APIRouter, HTTPException

from app.models.query import Citation, QueryRequest, QueryResponse, SourceChunk
from app.services.citation.builder import build_citation
from app.services.retrieval.pipeline import get_query_pipeline
from app.utils.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    # user_id is taken from the request payload; repositories filter on it for isolation.
    settings = get_settings()
    use_agent = request.use_agent if request.use_agent is not None else settings.agent.enabled
    style = request.agent_style or settings.agent.style
    try:
        if use_agent and style in ("tools", "deep"):
            return await _run_tool_agent_query(request, style)
        if use_agent:
            return await _run_agent_query(request)
        return await get_query_pipeline().run(request)
    except (ValueError, NotImplementedError) as exc:
        # Client/config errors — handled, non-breaking.
        logger.warning("Query rejected: %s", exc)
        status = 400 if isinstance(exc, ValueError) else 501
        raise HTTPException(status_code=status, detail=str(exc))
    except Exception as exc:
        # Unexpected — breaking; log with traceback and surface a 500.
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error processing query")


async def _run_agent_query(request: QueryRequest) -> QueryResponse:
    """Run the LangGraph agent (rewrite/decompose/grade-loop) for a stateless /query call."""
    from app.models.agent import AgentState
    from app.services.agent.graph import run_agent

    logger.info("Routing query through the agent")
    state = AgentState(
        query=request.query,
        user_id=request.user_id,
        mode=request.mode,
        filters=request.filters,
    )
    result = await run_agent(state)

    source_chunks = [SourceChunk(**chunk) for chunk in result.citations]
    scores = [chunk.score for chunk in source_chunks]
    confidence = min(sum(scores) / len(scores), 1.0) if scores else 0.0

    timings = dict(result.app_timings)
    timings["total"] = round(sum(timings.values()), 1)

    return QueryResponse(
        answer=result.answer or "I was unable to generate an answer.",
        citations=Citation(source_chunks=source_chunks, confidence=confidence),
        mode_used=result.mode or request.mode or get_settings().retrieval.default_mode,
        session_id=request.session_id,
        app_timings=timings,
    )


def _dedupe_by_id(chunks: list) -> list:
    seen: set = set()
    unique = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id"))
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk)
    return unique


async def _run_tool_agent_query(request: QueryRequest, style: str) -> QueryResponse:
    """Run the tool-calling agent (style="tools") or deepagents (style="deep") for a /query call."""
    from app.services.agent.tool_agent import run_tool_agent

    mode = request.mode or get_settings().retrieval.default_mode
    start = time.perf_counter()
    answer, chunks = await run_tool_agent(
        request.query, request.user_id, mode, request.filters, deep=(style == "deep")
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)
    citation = build_citation(_dedupe_by_id(chunks))
    return QueryResponse(
        answer=answer or "I was unable to generate an answer.",
        citations=citation,
        mode_used=f"{style}:{mode}",
        session_id=request.session_id,
        app_timings={"tool_agent": elapsed_ms, "total": elapsed_ms},
    )
