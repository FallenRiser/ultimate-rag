"""
ReAct tool-calling agent (deepagents-style). The LLM is the orchestrator: it calls
retrieval tools as many times as it needs, then answers. Selected via agent.style="tools"
(or per-request agent_style). Needs a tool-calling-capable model — point llm.provider at a
strong model for best results.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.services.agent.graph import _get_services
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a retrieval agent answering questions over a private knowledge base. "
    "Use the tools to search as many times as needed — rewrite or decompose the question "
    "yourself between searches when it helps you find better evidence. "
    "Retrieval modes: 'semantic' (always available), 'bm25'/'hybrid' (keyword + semantic), "
    "'graph'/'hybrid_graph' (entity knowledge graph). If a mode errors, fall back to 'semantic'. "
    "When you have enough evidence, answer concisely using ONLY what the tools returned. "
    "If the knowledge base does not contain the answer, say so plainly."
)


def _build_chat_model():
    """LangChain chat model over the OpenAI-compatible API (covers OpenAI, vLLM, and Ollama)."""
    from langchain_openai import ChatOpenAI

    cfg = get_settings().llm
    if cfg.provider == "ollama":
        base = cfg.base_url.rstrip("/")
        base_url = base if base.endswith("/v1") else f"{base}/v1"
        api_key = "ollama"  # ignored by Ollama, but the client requires a value
    else:
        base_url = os.getenv("OPENAI_BASE_URL") or cfg.base_url
        api_key = os.getenv("OPENAI_API_KEY") or cfg.api_key or "missing"

    return ChatOpenAI(model=cfg.model, base_url=base_url, api_key=api_key, temperature=cfg.temperature)


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No results."
    blocks = []
    for chunk in chunks:
        text = chunk.get("payload", {}).get("text", "")[:500]
        blocks.append(f"[{chunk.get('id')}] {text}")
    return "\n\n".join(blocks)


def _make_tools(user_id: str, filters: Dict[str, Any], collector: List[Dict[str, Any]]) -> list:
    """Build per-request tools. They capture user_id (isolation) and append every retrieved
    chunk to `collector` so the route can build citations afterwards."""
    from langchain_core.tools import tool

    svc = _get_services()

    @tool
    async def search_documents(query: str, mode: str = "semantic", top_k: int = 10) -> str:
        """Search the knowledge base. mode is one of: semantic, bm25, hybrid, graph, hybrid_graph.
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
        chunks = await svc.retrieval.vector_db.list_by_document(document_id, user_id)
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
    prompt = f"{_SYSTEM}\nPreferred retrieval mode: {mode}. Use it first; fall back to 'semantic' if it errors."
    agent = _build_agent(model, tools, prompt, deep)

    # Deep agents take extra steps for planning/sub-agents, so allow a larger budget.
    recursion_limit = max_calls * (3 if deep else 2) + (10 if deep else 1)

    messages = list(history or []) + [{"role": "user", "content": query}]
    logger.info("Tool agent running (style=%s, mode=%s, max_tool_calls=%d)",
                "deep" if deep else "tools", mode, max_calls)
    result = await agent.ainvoke({"messages": messages}, config={"recursion_limit": recursion_limit})

    answer = result["messages"][-1].content
    logger.info("Tool agent done: %d chunks retrieved across tool calls", len(collector))
    return answer, collector
