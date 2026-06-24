from typing import List, Optional

from openai import AsyncOpenAI

from app.services.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbeddings(BaseEmbeddingProvider):
    def __init__(self, api_key: Optional[str], base_url: Optional[str], model: str, dim: int):
        # api_key/base_url None → AsyncOpenAI falls back to OPENAI_API_KEY / OPENAI_BASE_URL env.
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> List[float]:
        results = await self.embed_documents([text])
        return results[0]
