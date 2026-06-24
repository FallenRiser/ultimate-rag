import uuid
from typing import Any, Dict, List

import tiktoken

from app.models.document import Chunk
from app.services.chunking.base import BaseChunker

_ENCODING = "cl100k_base"


class FixedChunker(BaseChunker):
    """Fixed-size token windows with overlap — no respect for sentence/paragraph boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = tiktoken.get_encoding(_ENCODING)

    async def chunk(
        self, text: str, document_id: str, version_id: str, metadata: Dict[str, Any]
    ) -> List[Chunk]:
        tokens = self._enc.encode(text)
        step = max(1, self.chunk_size - self.chunk_overlap)

        result = []
        ordinal = 0
        for start in range(0, len(tokens), step):
            window = tokens[start: start + self.chunk_size]
            if not window:
                break
            chunk_text = self._enc.decode(window).strip()
            if not chunk_text:
                continue
            result.append(Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                version_id=version_id,
                ordinal=ordinal,
                text=chunk_text,
                metadata=dict(metadata),
                token_count=len(window),
            ))
            ordinal += 1
        return result
