from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    role: str                            # user | assistant
    content: str
    citations: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"             # tenant id; supplied in the payload (no auth layer)
    session_id: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    agent_style: Optional[str] = None    # None → follow config agent.style; "graph" | "tools"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    message: ChatMessage
    citations: Optional[Dict[str, Any]] = None
    app_timings: Dict[str, float] = Field(default_factory=dict)  # per-stage milliseconds
