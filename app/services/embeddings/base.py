from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...
