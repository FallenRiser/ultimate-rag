from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"             # tenant id; supplied in the payload (no auth layer)
    session_id: Optional[str] = None
    mode: Optional[str] = None           # overrides config default if set
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: Optional[int] = None
    use_agent: Optional[bool] = None     # None → follow config agent.enabled; True/False overrides
    agent_style: Optional[str] = None    # None → follow config agent.style; "graph" | "tools"


class QueryEntities(BaseModel):
    """LLM-extracted entity names from a user query, used to seed graph retrieval."""
    entities: List[str] = Field(default_factory=list)


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page: Optional[int] = None
    score: float
    retrieved_by: List[str] = Field(default_factory=list)  # e.g. ["graph"], ["dense","bm25"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source_chunks: List[SourceChunk]
    confidence: float                    # derived from rerank scores + grounding check


class QueryResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    answer: str
    citations: Citation
    mode_used: str
    session_id: Optional[str] = None
    app_timings: Dict[str, float] = Field(default_factory=dict)  # per-stage milliseconds
