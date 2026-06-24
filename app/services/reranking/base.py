from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_n: int
    ) -> List[Dict[str, Any]]: ...
