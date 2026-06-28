from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Working state threaded through the LangGraph agent nodes."""
    query: str
    user_id: str
    session_id: Optional[str] = None
    mode: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)            # working set: explicit + inferred
    explicit_filters: Dict[str, Any] = Field(default_factory=dict)   # request filters only (never inferred)
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


# Structured LLM outputs used by the agent nodes

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


class FilterCondition(BaseModel):
    key: str
    values: List[str] = Field(default_factory=list)   # OR within the key (multi-value match)


class InferredFilters(BaseModel):
    conditions: List[FilterCondition] = Field(default_factory=list)
