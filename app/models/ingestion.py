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
    user_id: str = "default"                            # tenant id; supplied in the payload
    sync: bool = False                                  # run inline vs. background task
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parser_options: Dict[str, Any] = Field(default_factory=dict)


class IngestBytesItem(BaseModel):
    filename: str
    content_base64: str


class IngestBytesRequest(BaseModel):
    files: List[IngestBytesItem]
    user_id: str = "default"                            # tenant id; supplied in the payload
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


# Structured LLM outputs produced during ingestion

class AttributePair(BaseModel):
    """One free-form, document-type-specific metadata field. Modelled as a key/value pair
    (not an open dict) because structured-output is far more reliable across LLM providers."""
    key: str
    value: str


class DocumentMetadata(BaseModel):
    # Universal core — applies to almost any document, always extracted.
    title: str = ""
    author: str = ""
    doc_type: str = ""          # e.g. report, 10-Q, article, email, contract, judgment
    summary: str = ""           # 1-2 sentence abstract
    language: str = ""          # language name or ISO code
    created_date: str = ""      # the document's own date if stated (ISO 8601 if possible)
    # Free-form, type-specific (e.g. genre, section, priority, jurisdiction). Drives filtering.
    attributes: List[AttributePair] = []


class ChunkAttributes(BaseModel):
    """Section-level (per-chunk) free-form attributes. Same shape as document attributes."""
    attributes: List[AttributePair] = []


class MetadataKey(BaseModel):
    """A discovered free-form attribute key for a tenant — the convergence catalog. Fed back
    into extraction (reuse existing keys) and into auto-filter (what's filterable)."""
    key: str
    value_samples: List[str] = []
    high_cardinality: bool = False


class ExtractedEntity(BaseModel):
    name: str                   # display name; the graph node id is derived from it (cross-doc merge)
    type: str = "Entity"
    description: str = ""        # 1-2 sentences; embedded for semantic matching and fed to the LLM


class ExtractedRelation(BaseModel):
    """A directed relationship between two extracted entities, referenced by name."""
    source: str
    target: str
    type: str = "related_to"    # short snake_case verb phrase, e.g. founded, acquired, located_in
    description: str = ""        # how the two entities relate, in one short clause


class ExtractedGraph(BaseModel):
    """One LLM call per chunk yields the chunk's local knowledge graph (GraphRAG-style)."""
    entities: List[ExtractedEntity] = []
    relations: List[ExtractedRelation] = []


class CommunitySummary(BaseModel):
    """LLM report for a detected community of entities (powers global/thematic queries)."""
    title: str = ""
    summary: str = ""
