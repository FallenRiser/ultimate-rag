import logging
import math
from typing import List, Optional

import tiktoken
from langchain_ollama import OllamaEmbeddings as _LangchainOllamaEmbeddings

from app.services.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_ENCODING = "cl100k_base"


class OllamaEmbeddings(BaseEmbeddingProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        max_input_tokens: Optional[int] = None,
    ):
        self.client = _LangchainOllamaEmbeddings(model=model, base_url=base_url.rstrip("/"))
        self.model = model
        self._dim = dim
        self.max_input_tokens = max_input_tokens
        self._enc = tiktoken.get_encoding(_ENCODING) if max_input_tokens else None

    @property
    def dim(self) -> int:
        return self._dim

    def _truncate(self, text: str) -> str:
        # Hard client-side cap. Small-context models (mxbai=512) reject longer inputs and
        # Ollama's server-side truncate is unreliable on its new engine. This caps the text
        # that gets EMBEDDED only; the stored chunk text is untouched. Chunk size should be
        # aligned to this cap so it rarely fires — when it does, the chunk's tail is excluded
        # from its vector, which silently hurts recall, so we log it.
        if not self.max_input_tokens:
            return text
        tokens = self._enc.encode(text)
        if len(tokens) <= self.max_input_tokens:
            return text
        logger.warning(
            "Embedding input truncated: %d -> %d tokens. The chunk tail is excluded from its "
            "vector — lower chunking.chunk_size or raise embeddings.max_input_tokens to fix.",
            len(tokens), self.max_input_tokens,
        )
        return self._enc.decode(tokens[: self.max_input_tokens])

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = [self._truncate(text) for text in texts]
        embeddings = await self.client.aembed_documents(inputs)

        # Some models (e.g. bge-m3 with flash attention on older GPUs) emit NaN — fail loudly
        # instead of poisoning the vector store with garbage vectors.
        if embeddings and any(math.isnan(value) for value in embeddings[0]):
            raise RuntimeError(
                f"Ollama model {self.model!r} returned NaN embeddings. "
                "Restart `ollama serve` with OLLAMA_FLASH_ATTENTION=0, or use a "
                "different embedding model (e.g. mxbai-embed-large, nomic-embed-text)."
            )
        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        results = await self.embed_documents([text])
        return results[0]
