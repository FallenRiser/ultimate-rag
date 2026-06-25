from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    user_id: str
    source: str
    mime_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UploadedFile(BaseModel):
    filename: str
    path: str
    size: int


class IngestByFilenameRequest(BaseModel):
    version: str
    filenames: List[str]
    sync: bool = False                                  # run inline vs. background task
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parser_options: Dict[str, Any] = Field(default_factory=dict)


class IngestBytesItem(BaseModel):
    filename: str
    content_base64: str


class IngestBytesRequest(BaseModel):
    files: List[IngestBytesItem]
    sync: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parser_options: Dict[str, Any] = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    document_id: str
    task_id: str
    status: str = "queued"


class IngestionStatus(BaseModel):
    document_id: str
    status: str
    progress: float = 0.0
    error: Optional[str] = None
