from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import httpx

from app.services.parsing.base import BaseDocumentParser, ParsedDocument
from app.utils.config import DoclingSettings


def _encode(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()       # docling-serve expects "true"/"false"
    return str(value)


def _to_multipart(form: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Flatten the options dict into multipart fields; list values become repeated keys."""
    fields: List[Tuple[str, str]] = []
    for key, value in form.items():
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                fields.append((key, _encode(item)))
        else:
            fields.append((key, _encode(value)))
    return fields


class DoclingParser(BaseDocumentParser):
    """Client for a running docling-serve instance. Options come from config and can be
    overridden per request via the ingestion API's `parser_options`."""

    def __init__(self, settings: DoclingSettings):
        self.settings = settings
        self.url = settings.url.rstrip("/")

    def supports(self, mime_type: str) -> bool:
        return True  # docling handles all document types

    def _build_options(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = self.settings
        options: Dict[str, Any] = {
            "to_formats": cfg.to_formats,
            "from_formats": cfg.from_formats,
            "do_ocr": cfg.do_ocr,
            "force_ocr": cfg.force_ocr,
            "ocr_engine": cfg.ocr_engine,
            "ocr_lang": cfg.ocr_lang,
            "pdf_backend": cfg.pdf_backend,
            "table_mode": cfg.table_mode,
            "do_table_structure": cfg.do_table_structure,
            "image_export_mode": cfg.image_export_mode,
            "abort_on_error": cfg.abort_on_error,
            "return_as_file": cfg.return_as_file,
            "document_timeout": cfg.document_timeout,
        }
        options.update(overrides or {})   # caller (API) overrides win
        return options

    async def parse(
        self,
        file: BinaryIO,
        mime_type: str,
        filename: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument:
        merged = self._build_options(options)
        file_bytes = file.read()

        # Give the HTTP client headroom over docling's own per-document timeout.
        doc_timeout = merged.get("document_timeout") or 0
        read_timeout = max(300.0, float(doc_timeout) + 60.0)

        async with httpx.AsyncClient(timeout=read_timeout) as client:
            response = await client.post(
                f"{self.url}/v1/convert/file",
                data=_to_multipart(merged),
                files={"files": (filename, file_bytes, mime_type)},
            )
            response.raise_for_status()
            body = response.json()

        document = body.get("document", body)
        text = document.get("md_content") or document.get("text_content") or ""
        return ParsedDocument(
            text=text,
            metadata={"filename": filename, "mime_type": mime_type, "parser": "docling"},
        )
