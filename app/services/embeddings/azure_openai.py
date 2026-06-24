from typing import List

from openai import AsyncAzureOpenAI

from app.services.embeddings.base import BaseEmbeddingProvider


class AzureOpenAIEmbeddings(BaseEmbeddingProvider):
    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str, dim: int):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.deployment = deployment
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(model=self.deployment, input=texts)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> List[float]:
        results = await self.embed_documents([text])
        return results[0]
