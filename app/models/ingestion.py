from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    user_id: str
    source: str
    mime_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    document_id: str
    task_id: str
    status: str = "queued"


class IngestionStatus(BaseModel):
    document_id: str
    status: str
    progress: float = 0.0
    error: Optional[str] = None
