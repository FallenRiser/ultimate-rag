import logging
from typing import Any, Dict, List

import httpx

from app.services.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):
    """BGE-reranker-v2-m3 via a Text Embeddings Inference (TEI) HTTP endpoint."""

    def __init__(self, endpoint: str, model: str, top_n: int = 8):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.top_n = top_n

    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_n: int
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        texts = [doc["payload"]["text"] for doc in documents]

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.endpoint}/rerank",
                json={"query": query, "texts": texts, "raw_scores": False},
            )
            response.raise_for_status()

        scores = response.json()  # [{"index": int, "score": float}, ...]
        for item in scores:
            documents[item["index"]]["rerank_score"] = item["score"]

        ranked = sorted(documents, key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        ranked = ranked[:top_n]
        logger.debug(
            "Reranked %d -> %d; top score=%.4f",
            len(documents), len(ranked), ranked[0].get("rerank_score", 0.0) if ranked else 0.0,
        )
        return ranked
