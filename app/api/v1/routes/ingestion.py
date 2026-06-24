import json
import mimetypes
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_current_user
from app.models.ingestion import IngestionResponse, IngestionStatus
from app.repositories.relational.database import get_engine
from app.repositories.relational.documents import get_document
from app.services.ingestion.service import IngestionService
from app.services.parsing.factory import create_parser
from app.utils.hashing import sha256_bytes

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _resolve_mime(file: UploadFile) -> str:
    return file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"


def _parse_metadata(metadata: Optional[str]) -> dict:
    """Parse the optional metadata form field (a JSON object string) into a flat dict.
    Values become filterable at query time (e.g. {"category": "finance", "year": "2024"})."""
    if not metadata:
        return {}
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be a valid JSON object")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return parsed


async def _read_and_register(file: UploadFile, user_id: str):
    mime_type = _resolve_mime(file)
    parser = create_parser()
    if not parser.supports(mime_type):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {mime_type!r}")

    content = await file.read()
    service = IngestionService()
    try:
        doc, dedup = await service.register(
            content=content,
            filename=file.filename or "upload",
            mime_type=mime_type,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return service, doc, dedup, content, mime_type


@router.post("", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
) -> IngestionResponse:
    """Async ingest — registers the document and processes it in a background task.
    `metadata` is an optional JSON object whose keys become filterable at query time."""
    extra_metadata = _parse_metadata(metadata)
    service, doc, dedup, content, mime_type = await _read_and_register(file, user_id)
    if dedup:
        return dedup

    background_tasks.add_task(
        service.process,
        document_id=doc.id,
        user_id=user_id,
        content=content,
        filename=doc.source,
        mime_type=mime_type,
        content_hash=sha256_bytes(content),
        extra_metadata=extra_metadata,
    )
    return IngestionResponse(document_id=doc.id, task_id=doc.id, status="queued")


@router.post("/sync", response_model=IngestionResponse)
async def ingest_document_sync(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
) -> IngestionResponse:
    """Sync ingest — runs the full pipeline inline and returns the final status.
    `metadata` is an optional JSON object whose keys become filterable at query time."""
    extra_metadata = _parse_metadata(metadata)
    service, doc, dedup, content, mime_type = await _read_and_register(file, user_id)
    if dedup:
        return dedup

    status = await service.process(
        document_id=doc.id,
        user_id=user_id,
        content=content,
        filename=doc.source,
        mime_type=mime_type,
        content_hash=sha256_bytes(content),
        extra_metadata=extra_metadata,
    )
    return IngestionResponse(document_id=doc.id, task_id="sync", status=status.value)


@router.get("/status/{document_id}", response_model=IngestionStatus)
async def ingestion_status(
    document_id: str,
    user_id: str = Depends(get_current_user),
) -> IngestionStatus:
    engine = get_engine()
    doc = await get_document(engine, document_id, user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    progress = 1.0 if doc.status.value == "ready" else (0.5 if doc.status.value == "processing" else 0.0)
    return IngestionStatus(document_id=document_id, status=doc.status.value, progress=progress)
