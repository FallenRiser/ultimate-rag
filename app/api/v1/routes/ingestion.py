import base64
import binascii
import json
import mimetypes
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_current_user
from app.models.ingestion import (
    IngestBytesRequest,
    IngestByFilenameRequest,
    IngestionResponse,
    IngestionStatus,
    UploadedFile,
)
from app.repositories.relational.database import get_engine
from app.repositories.relational.documents import get_document
from app.repositories.storage.files import read_file, save_file
from app.services.ingestion.service import IngestionService
from app.services.parsing.factory import create_parser
from app.utils.hashing import sha256_bytes

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _mime_for(filename: str, fallback: Optional[str] = None) -> str:
    return fallback or mimetypes.guess_type(filename or "")[0] or "application/octet-stream"


def _parse_json_object(raw: Optional[str], field: str) -> dict:
    """Parse an optional JSON-object form field into a dict (empty if absent)."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"{field} must be a valid JSON object")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail=f"{field} must be a JSON object")
    return parsed


async def _ingest_content(
    content: bytes,
    filename: str,
    user_id: str,
    extra_metadata: dict,
    parser_options: dict,
    background_tasks: Optional[BackgroundTasks],
) -> IngestionResponse:
    """Shared ingest core: validate → register → process (background or inline).
    Passing `background_tasks` queues processing; passing None runs it inline."""
    mime_type = _mime_for(filename)
    parser = create_parser()
    if not parser.supports(mime_type):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {mime_type!r}")

    service = IngestionService()
    try:
        doc, dedup = await service.register(
            content=content, filename=filename, mime_type=mime_type, user_id=user_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if dedup:
        return dedup

    task_kwargs = dict(
        document_id=doc.id,
        user_id=user_id,
        content=content,
        filename=doc.source,
        mime_type=mime_type,
        content_hash=sha256_bytes(content),
        extra_metadata=extra_metadata,
        parser_options=parser_options,
    )
    if background_tasks is not None:
        background_tasks.add_task(service.process, **task_kwargs)
        return IngestionResponse(document_id=doc.id, task_id=doc.id, status="queued")

    status = await service.process(**task_kwargs)
    return IngestionResponse(document_id=doc.id, task_id="sync", status=status.value)


# --- 1. Upload only: save the raw file under {user_id}/{version}, no ingestion --------
@router.post("/upload", response_model=List[UploadedFile])
async def upload_files(
    version: str = Form(...),
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
) -> List[UploadedFile]:
    """Store uploaded files verbatim under {storage_dir}/{user_id}/{version}/. Ingest later
    with POST /ingest/by-filename using the same version + filenames."""
    saved: List[UploadedFile] = []
    for file in files:
        content = await file.read()
        path = await run_in_threadpool(save_file, user_id, version, file.filename or "upload", content)
        saved.append(UploadedFile(filename=path.name, path=str(path), size=len(content)))
    return saved


# --- 2. Ingest previously-uploaded files by version + filename -----------------------
@router.post("/by-filename", response_model=List[IngestionResponse])
async def ingest_by_filename(
    request: IngestByFilenameRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> List[IngestionResponse]:
    """Ingest files already saved via /ingest/upload, located by version + filename."""
    results: List[IngestionResponse] = []
    for filename in request.filenames:
        try:
            content = await run_in_threadpool(read_file, user_id, request.version, filename)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found: {filename} (version {request.version})")
        results.append(await _ingest_content(
            content=content,
            filename=filename,
            user_id=user_id,
            extra_metadata=request.metadata,
            parser_options=request.parser_options,
            background_tasks=None if request.sync else background_tasks,
        ))
    return results


# --- 3. Ingest raw file bytes (base64) directly, no storage -------------------------
@router.post("/bytes", response_model=List[IngestionResponse])
async def ingest_bytes(
    request: IngestBytesRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> List[IngestionResponse]:
    """Ingest files supplied as base64-encoded bytes in the request body."""
    results: List[IngestionResponse] = []
    for item in request.files:
        try:
            content = base64.b64decode(item.content_base64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid base64 for {item.filename}")
        results.append(await _ingest_content(
            content=content,
            filename=item.filename,
            user_id=user_id,
            extra_metadata=request.metadata,
            parser_options=request.parser_options,
            background_tasks=None if request.sync else background_tasks,
        ))
    return results


@router.post("", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    parser_options: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
) -> IngestionResponse:
    """Async ingest — registers the document and processes it in a background task.
    `metadata`: JSON object, keys become filterable at query time.
    `parser_options`: JSON object overriding docling-serve settings for this document."""
    content = await file.read()
    return await _ingest_content(
        content=content,
        filename=file.filename or "upload",
        user_id=user_id,
        extra_metadata=_parse_json_object(metadata, "metadata"),
        parser_options=_parse_json_object(parser_options, "parser_options"),
        background_tasks=background_tasks,
    )


@router.post("/sync", response_model=IngestionResponse)
async def ingest_document_sync(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    parser_options: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
) -> IngestionResponse:
    """Sync ingest — runs the full pipeline inline and returns the final status.
    `metadata`: JSON object, keys become filterable at query time.
    `parser_options`: JSON object overriding docling-serve settings for this document."""
    content = await file.read()
    return await _ingest_content(
        content=content,
        filename=file.filename or "upload",
        user_id=user_id,
        extra_metadata=_parse_json_object(metadata, "metadata"),
        parser_options=_parse_json_object(parser_options, "parser_options"),
        background_tasks=None,
    )


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
