from typing import List, Tuple
from urllib.parse import parse_qs, urlparse

from langchain_openai import AzureOpenAIEmbeddings as _LangchainAzureEmbeddings

from app.services.embeddings.base import BaseEmbeddingProvider


def parse_azure_url(url: str) -> Tuple[str, str, str]:
    """Split a full Azure deployment URL into (endpoint, deployment, api_version).
    Expects: https://<res>.openai.azure.com/openai/deployments/<dep>/embeddings?api-version=<ver>"""
    parsed = urlparse(url)
    endpoint = f"{parsed.scheme}://{parsed.netloc}"
    parts = parsed.path.strip("/").split("/")
    deployment = parts[parts.index("deployments") + 1] if "deployments" in parts else ""
    api_version = parse_qs(parsed.query).get("api-version", [""])[0]
    return endpoint, deployment, api_version


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
