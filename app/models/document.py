from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    source: str                           # filename or URL
    mime_type: str
    status: DocumentStatus = DocumentStatus.pending
    current_version_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentVersion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    version_no: int
    content_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    version_id: str
    ordinal: int
    text: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0


class DocumentChunk(BaseModel):
    """A chunk as stored in the vector store (returned by the list-chunks API)."""
    chunk_id: str
    document_id: str
    ordinal: Optional[int] = None
    page: Optional[int] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
