"""
ReAct tool-calling agent (deepagents-style). The LLM is the orchestrator: it calls
retrieval tools as many times as it needs, then answers. Selected via agent.style="tools"
(or per-request agent_style). Needs a tool-calling-capable model — point llm.provider at a
strong model for best results.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.observability.tracing import log_chunks_to_trace, request_trace
from app.prompts.agent import TOOL_AGENT_SYSTEM
from app.services.agent.graph import _get_services
from app.utils.config import get_settings

logger = logging.getLogger(__name__)


def _build_chat_model():
    """LangChain chat model over the OpenAI-compatible API (covers OpenAI, vLLM, and Ollama)."""
    from langchain_openai import ChatOpenAI

    cfg = get_settings().llm
    if cfg.provider == "ollama":
        base = cfg.base_url.rstrip("/")
        base_url = base if base.endswith("/v1") else f"{base}/v1"
        api_key = "ollama"  # ignored by Ollama, but the client requires a value
    elif cfg.provider == "vllm":
        base_url = os.getenv("OPENAI_BASE_URL") or cfg.base_url
        api_key = os.getenv("OPENAI_API_KEY") or cfg.api_key or "missing"
    else:  # openai → the real OpenAI API; ignore llm.base_url (defaults to the Ollama URL) unless
           # an OpenAI endpoint is explicitly given via env. base_url=None → ChatOpenAI uses OpenAI.
        base_url = os.getenv("OPENAI_BASE_URL") or None
        api_key = os.getenv("OPENAI_API_KEY") or cfg.api_key or "missing"

    return ChatOpenAI(model=cfg.model, base_url=base_url, api_key=api_key, temperature=cfg.temperature)


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No results."
    blocks = []
    for chunk in chunks:
        text = chunk.get("payload", {}).get("text", "")
        blocks.append(f"[{chunk.get('id')}] {text}")
    return "\n\n".join(blocks)


def _message_text(content: Any) -> str:
    """LangChain message content is usually a string but can be a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        )
    return str(content)


def _make_tools(user_id: str, filters: Dict[str, Any], collector: List[Dict[str, Any]]) -> list:
    """Build per-request tools. They capture user_id (isolation) and append every retrieved
    chunk to `collector` so the route can build citations afterwards."""
    from langchain_core.tools import tool

    svc = _get_services()

    @tool
    async def search_documents(query: str, mode: str = "semantic", top_k: int = 10) -> str:
        """Search the knowledge base. mode is one of: semantic, bm25, hybrid, graph, hybrid_graph,
        graph_global (broad/thematic questions about the whole knowledge base).
        Returns matching chunks as "[chunk_id] text". Use semantic if unsure."""
        try:
            query_vector = await svc.embedding.embed_query(query)
            chunks = await svc.retrieval.retrieve(
                query=query, query_vector=query_vector, user_id=user_id,
                mode=mode, top_k=top_k, filters=filters,
            )
        except (ValueError, NotImplementedError) as exc:
            return f"Search error: {exc} Try mode='semantic'."
        collector.extend(chunks)
        return _format_chunks(chunks)

    @tool
    async def list_document_chunks(document_id: str) -> str:
        """Return all chunks of one document (by its document_id) in order — use to read a
        document fully once you have identified it from a search result's metadata."""
        try:
            chunks = await svc.retrieval.vector_db.list_by_document(document_id, user_id)
        except Exception as exc:
            # Non-breaking: return a recoverable message so the LLM can continue.
            logger.warning("list_document_chunks failed for %s: %s", document_id, exc)
            logger.debug("list_document_chunks failure (doc=%s)", document_id, exc_info=True)
            return f"Could not read document {document_id}: {exc}"
        chunks.sort(key=lambda c: c.get("payload", {}).get("metadata", {}).get("ordinal", 0))
        collector.extend(chunks)
        return _format_chunks(chunks)

    return [search_documents, list_document_chunks]


def _build_agent(model: Any, tools: list, system_prompt: str, deep: bool) -> Any:
    """Build the orchestrator. deep=True uses deepagents (planner + sub-agents + scratchpad);
    deep=False uses a plain ReAct tool-calling agent."""
    if deep:
        from deepagents import create_deep_agent
        return create_deep_agent(model, tools, system_prompt=system_prompt)
    try:
        from langchain.agents import create_agent
        return create_agent(model, tools, system_prompt=system_prompt)
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(model, tools, prompt=system_prompt)


async def run_tool_agent(
    query: str,
    user_id: str,
    mode: str,
    filters: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None,
    deep: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the tool-calling agent. Returns (answer, retrieved_chunks) — chunks build citations."""
    settings = get_settings()
    max_calls = settings.agent.tools.max_tool_calls

    collector: List[Dict[str, Any]] = []
    model = _build_chat_model()
    tools = _make_tools(user_id, filters, collector)
    prompt = f"{TOOL_AGENT_SYSTEM}\nPreferred retrieval mode: {mode}. Use it first; fall back to 'semantic' if it errors."
    agent = _build_agent(model, tools, prompt, deep)

    # Deep agents take extra steps for planning/sub-agents, so allow a larger budget.
    recursion_limit = max_calls * (3 if deep else 2) + (10 if deep else 1)

    from langgraph.errors import GraphRecursionError

    messages = list(history or []) + [{"role": "user", "content": query}]
    logger.info("Tool agent running (style=%s, mode=%s, max_tool_calls=%d)",
                "deep" if deep else "tools", mode, max_calls)

    # One root span per request so every tool's embedding/LLM call nests in one trace.
    answer = ""
    with request_trace("rag_tool_agent", {"query": query, "user_id": user_id, "mode": mode}) as span:
        try:
            result = await agent.ainvoke({"messages": messages}, config={"recursion_limit": recursion_limit})
            answer = _message_text(result["messages"][-1].content)
        except GraphRecursionError as exc:
            # Non-breaking: ran past the step budget. Return a best-effort answer — the route
            # still gets whatever chunks the tools gathered. Other errors propagate to the route.
            logger.warning("Tool agent hit recursion_limit=%d; returning best-effort answer: %s", recursion_limit, exc)
            logger.debug("Tool agent recursion limit (query=%r)", query, exc_info=True)
            answer = "I couldn't fully complete the search within the allowed steps. Try narrowing the question."
        log_chunks_to_trace(span, collector)
        if span is not None:
            span.set_outputs({"answer": answer})

    logger.info("Tool agent done: %d chunks retrieved across tool calls", len(collector))
    return answer, collector
