from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import httpx

from app.services.parsing.base import BaseDocumentParser, ParsedDocument
from app.utils.config import DoclingSettings


def _encode(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()       # docling-serve expects "true"/"false"
    return str(value)


# docling treats these as mutually exclusive (per its API docs); at most one per group.
_MUTUALLY_EXCLUSIVE: List[Tuple[str, ...]] = [
    ("picture_description_local", "picture_description_api"),
    ("vlm_pipeline_model", "vlm_pipeline_model_local", "vlm_pipeline_model_api"),
]


def _check_mutually_exclusive(options: Dict[str, Any]) -> None:
    for group in _MUTUALLY_EXCLUSIVE:
        conflicting = [key for key in group if options.get(key) not in (None, "")]
        if len(conflicting) > 1:
            raise ValueError(f"docling options {conflicting} are mutually exclusive; set only one")


def _to_form(form: Dict[str, Any]) -> Dict[str, Any]:
    """Build httpx multipart form fields: drop None, bools → "true"/"false", list values
    kept as lists so httpx emits them as repeated keys. Must be a dict (not a list of
    tuples) — a list makes httpx build a sync stream that AsyncClient rejects."""
    fields: Dict[str, Any] = {}
    for key, value in form.items():
        if value is None:
            continue
        if isinstance(value, list):
            fields[key] = [_encode(item) for item in value]
        else:
            fields[key] = _encode(value)
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
            "target_type": cfg.target_type,
            "to_formats": cfg.to_formats,
            "from_formats": cfg.from_formats,
            "do_ocr": cfg.do_ocr,
            "force_ocr": cfg.force_ocr,
            "ocr_engine": cfg.ocr_engine,
            "ocr_lang": cfg.ocr_lang,
            "pdf_backend": cfg.pdf_backend,
            "table_mode": cfg.table_mode,
            "table_cell_matching": cfg.table_cell_matching,
            "do_table_structure": cfg.do_table_structure,
            "pipeline": cfg.pipeline,
            "image_export_mode": cfg.image_export_mode,
            "include_images": cfg.include_images,
            "images_scale": cfg.images_scale,
            "md_page_break_placeholder": cfg.md_page_break_placeholder,
            "page_range": cfg.page_range,
            "document_timeout": cfg.document_timeout,
            "abort_on_error": cfg.abort_on_error,
            "do_code_enrichment": cfg.do_code_enrichment,
            "do_formula_enrichment": cfg.do_formula_enrichment,
            "do_picture_classification": cfg.do_picture_classification,
            "do_chart_extraction": cfg.do_chart_extraction,
            "do_picture_description": cfg.do_picture_description,
            "picture_description_area_threshold": cfg.picture_description_area_threshold,
            "picture_description_local": cfg.picture_description_local,
            "picture_description_api": cfg.picture_description_api,
            "vlm_pipeline_model": cfg.vlm_pipeline_model,
            "vlm_pipeline_model_local": cfg.vlm_pipeline_model_local,
            "vlm_pipeline_model_api": cfg.vlm_pipeline_model_api,
        }
        options.update(overrides or {})   # caller (API) overrides win
        _check_mutually_exclusive(options)
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
                data=_to_form(merged),
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


if __name__ == "__main__":
    _check_mutually_exclusive({"do_ocr": True})  # no conflict
    _check_mutually_exclusive({"vlm_pipeline_model": "x"})  # one set is fine
    for bad in (
        {"picture_description_local": "a", "picture_description_api": "b"},
        {"vlm_pipeline_model": "a", "vlm_pipeline_model_api": "b"},
    ):
        try:
            _check_mutually_exclusive(bad)
            raise SystemExit(f"expected ValueError for {bad}")
        except ValueError:
            pass
    print("OK")
