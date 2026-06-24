import math
import re
import uuid
from typing import Any, Dict, List

import tiktoken

from app.models.document import Chunk
from app.services.chunking.base import BaseChunker

_ENCODING = "cl100k_base"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker(BaseChunker):
    """Splits on embedding similarity drops — a new chunk begins where the cosine
    similarity between consecutive sentences falls below `threshold`. A token cap
    bounds runaway chunks when many sentences stay similar."""

    def __init__(self, threshold: float = 0.6, max_tokens: int = 512):
        self.threshold = threshold
        self.max_tokens = max_tokens
        self._enc = tiktoken.get_encoding(_ENCODING)

    async def chunk(
        self, text: str, document_id: str, version_id: str, metadata: Dict[str, Any]
    ) -> List[Chunk]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return []

        from app.services.embeddings.factory import create_embedding_provider
        embedder = create_embedding_provider()
        vectors = await embedder.embed_documents(sentences)

        groups: List[List[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            similarity = _cosine(vectors[i - 1], vectors[i])
            current = groups[-1]
            over_cap = self._count(" ".join(current + [sentences[i]])) > self.max_tokens
            if similarity < self.threshold or over_cap:
                groups.append([sentences[i]])
            else:
                current.append(sentences[i])

        result = []
        for ordinal, group in enumerate(groups):
            chunk_text = " ".join(group).strip()
            if not chunk_text:
                continue
            result.append(Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                version_id=version_id,
                ordinal=ordinal,
                text=chunk_text,
                metadata=dict(metadata),
                token_count=self._count(chunk_text),
            ))
        return result

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
