from app.services.parsing.base import BaseDocumentParser
from app.utils.config import get_settings


def create_parser() -> BaseDocumentParser:
    settings = get_settings()
    backend = settings.parsing.backend

    if backend == "docling":
        from app.services.parsing.docling_client import DoclingParser
        return DoclingParser(settings=settings.parsing.docling)

    if backend == "custom":
        from app.services.parsing.custom_parser import CustomParser
        cfg = settings.parsing.custom
        return CustomParser(ocr_engine=cfg.ocr_engine, ocr_languages=cfg.ocr_languages)

    raise ValueError(f"Unknown parsing.backend: {backend!r}")
