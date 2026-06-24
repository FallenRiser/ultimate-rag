from typing import Optional

from app.services.reranking.base import BaseReranker
from app.utils.config import get_settings


def create_reranker() -> Optional[BaseReranker]:
    settings = get_settings()
    cfg = settings.reranker

    if not cfg.enabled:
        return None

    if cfg.provider == "bge":
        from app.services.reranking.bge import BGEReranker
        return BGEReranker(endpoint=cfg.endpoint or "", model=cfg.model, top_n=cfg.top_n)

    raise ValueError(f"Unknown reranker.provider: {cfg.provider!r}")
