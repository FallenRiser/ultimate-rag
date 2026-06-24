import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.deps import get_current_user
from app.models.chat import ChatMessage, ChatRequest, ChatResponse, ChatSession
from app.repositories.relational.database import get_engine
from app.repositories.relational.messages import add_message, list_recent_messages
from app.repositories.relational.sessions import (
    create_session,
    delete_session,
    get_session,
    list_sessions,
    touch_session,
)
from app.services.agent.graph import AgentState, run_agent
from app.utils.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


async def _load_history(engine: AsyncEngine, session_id: str, user_id: str) -> List[Dict[str, str]]:
    limit = get_settings().chat.max_turns * 2  # each turn is a user + assistant message
    messages = await list_recent_messages(engine, session_id, user_id, limit)
    return [{"role": m.role, "content": m.content} for m in messages]


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
) -> ChatResponse:
    engine = get_engine()

    # Resolve or create session
    session_id = request.session_id
    if session_id:
        session = await get_session(engine, session_id, user_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        await touch_session(engine, session_id)
    else:
        session = ChatSession(user_id=user_id, title=request.message[:60])
        await create_session(engine, session)
        session_id = session.id

    logger.info("Received chat message")
    logger.debug("Received chat message: %r user_id=%s session_id=%s", request.message, user_id, session_id)

    # Prior turns drive follow-up resolution (the agent contextualizes against them).
    history = await _load_history(engine, session_id, user_id)

    style = request.agent_style or get_settings().agent.style
    if style in ("tools", "deep"):
        answer, citations, timings = await _run_tool_agent_chat(request, user_id, history, style)
    else:
        answer, citations, timings = await _run_graph_agent_chat(request, user_id, session_id, history)

    # Persist the turn so the next message can use it as history.
    user_message = ChatMessage(session_id=session_id, role="user", content=request.message)
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=answer,
        citations={"source_chunks": citations},
    )
    await add_message(engine, user_message, user_id)
    await add_message(engine, assistant_message, user_id)

    logger.info("Chat complete: %s", timings)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        message=assistant_message,
        citations={"source_chunks": citations},
        app_timings=timings,
    )


async def _run_graph_agent_chat(
    request: ChatRequest, user_id: str, session_id: str, history: List[Dict[str, str]]
) -> tuple:
    """Fixed state-machine agent (rewrite/decompose/grade-loop) with conversation memory."""
    state = AgentState(
        query=request.message,
        user_id=user_id,
        session_id=session_id,
        filters=request.filters,
        history=history,
    )
    result = await run_agent(state)
    answer = result.answer or "I was unable to generate an answer."
    timings = dict(result.app_timings)
    timings["total"] = round(sum(timings.values()), 1)
    return answer, result.citations, timings


async def _run_tool_agent_chat(
    request: ChatRequest, user_id: str, history: List[Dict[str, str]], style: str
) -> tuple:
    """Tool-calling agent (style="tools") or deepagents (style="deep"). Citations come from
    whatever chunks its tools retrieved."""
    from app.services.agent.tool_agent import run_tool_agent
    from app.services.citation.builder import build_citation

    mode = get_settings().retrieval.default_mode
    answer, chunks = await run_tool_agent(
        request.message, user_id, mode, request.filters, history, deep=(style == "deep")
    )

    seen: set = set()
    unique = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id"))
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk)
    citation = build_citation(unique)
    citations = [chunk.model_dump() for chunk in citation.source_chunks]
    return answer or "I was unable to generate an answer.", citations, {}


@router.get("/sessions", response_model=list[ChatSession])
async def list_user_sessions(
    user_id: str = Depends(get_current_user),
) -> list[ChatSession]:
    return await list_sessions(get_engine(), user_id)


@router.delete("/sessions/{session_id}")
async def delete_user_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    deleted = await delete_session(get_engine(), session_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}
