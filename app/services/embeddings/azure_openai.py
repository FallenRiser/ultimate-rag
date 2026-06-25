from typing import List

from langchain_openai import AzureOpenAIEmbeddings as _LangchainAzureEmbeddings

from app.services.embeddings.base import BaseEmbeddingProvider


class AzureOpenAIEmbeddings(BaseEmbeddingProvider):
    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str, dim: int):
        self.client = _LangchainAzureEmbeddings(
            api_key=api_key or None,        # None → reads AZURE_OPENAI_API_KEY env
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
        )
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.client.aembed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        return await self.client.aembed_query(text)
