from typing import List, Optional

from langchain_openai import OpenAIEmbeddings as _LangchainOpenAIEmbeddings

from app.services.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbeddings(BaseEmbeddingProvider):
    def __init__(self, api_key: Optional[str], base_url: Optional[str], model: str, dim: int):
        # api_key/base_url None → reads OPENAI_API_KEY / OPENAI_BASE_URL env.
        self.client = _LangchainOpenAIEmbeddings(
            model=model, api_key=api_key or None, base_url=base_url or None
        )
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.client.aembed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        return await self.client.aembed_query(text)
