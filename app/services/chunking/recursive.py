import uuid
from typing import Any, Dict, List

import tiktoken

from app.models.document import Chunk
from app.services.chunking.base import BaseChunker

_SEPARATORS = ["\n\n", "\n", ". ", " "]
_ENCODING = "cl100k_base"


class RecursiveChunker(BaseChunker):
    """Splits on paragraph → line → sentence → word, then by token count as last resort."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = tiktoken.get_encoding(_ENCODING)

    async def chunk(
        self, text: str, document_id: str, version_id: str, metadata: Dict[str, Any]
    ) -> List[Chunk]:
        raw = self._split(text)
        result = []
        for i, chunk_text in enumerate(raw):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            result.append(Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                version_id=version_id,
                ordinal=i,
                text=chunk_text,
                metadata=dict(metadata),
                token_count=self._count(chunk_text),
            ))
        return result

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _tail(self, text: str) -> str:
        """Return the tail of text that fits within chunk_overlap tokens."""
        tokens = self._enc.encode(text)
        return self._enc.decode(tokens[-self.chunk_overlap:])

    def _split(self, text: str) -> List[str]:
        if self._count(text) <= self.chunk_size:
            return [text]

        for sep in _SEPARATORS:
            if sep not in text:
                continue

            parts = text.split(sep)
            chunks: List[str] = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if self._count(candidate) <= self.chunk_size:
                    current = candidate
                elif current:
                    chunks.append(current)
                    overlap = self._tail(current)
                    current = overlap + sep + part if self._count(overlap + sep + part) <= self.chunk_size else part
                else:
                    # Single part still too big — recurse with next separator
                    sub = self._split(part)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""

            if current.strip():
                chunks.append(current)

            return chunks if chunks else [text]

        # No separator fits — hard split at token boundary
        tokens = self._enc.encode(text)
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [
            self._enc.decode(tokens[i: i + self.chunk_size])
            for i in range(0, len(tokens), step)
            if tokens[i: i + self.chunk_size]
        ]
