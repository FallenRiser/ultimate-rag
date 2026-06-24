import math
from typing import List, Optional

import httpx
import tiktoken

from app.services.embeddings.base import BaseEmbeddingProvider

_ENCODING = "cl100k_base"


class OllamaEmbeddings(BaseEmbeddingProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        max_input_tokens: Optional[int] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dim
        self.max_input_tokens = max_input_tokens
        self._enc = tiktoken.get_encoding(_ENCODING) if max_input_tokens else None

    @property
    def dim(self) -> int:
        return self._dim

    def _truncate(self, text: str) -> str:
        # Hard client-side cap. Ollama's server-side truncate is unreliable on its
        # new engine, and small-context models (mxbai=512) reject longer inputs.
        if not self.max_input_tokens:
            return text
        tokens = self._enc.encode(text)
        if len(tokens) <= self.max_input_tokens:
            return text
        return self._enc.decode(tokens[: self.max_input_tokens])

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = [self._truncate(t) for t in texts]
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs, "truncate": True},
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]

        # Some models (e.g. bge-m3 with flash attention on older GPUs) emit NaN.
        # Fail loudly instead of poisoning the vector store with garbage vectors.
        if embeddings and any(math.isnan(v) for v in embeddings[0]):
            raise RuntimeError(
                f"Ollama model {self.model!r} returned NaN embeddings. "
                "Restart `ollama serve` with OLLAMA_FLASH_ATTENTION=0, or use a "
                "different embedding model (e.g. mxbai-embed-large, nomic-embed-text)."
            )
        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        results = await self.embed_documents([text])
        return results[0]
