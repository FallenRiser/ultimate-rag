from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.models.document import Chunk


class BaseChunker(ABC):
    @abstractmethod
    async def chunk(
        self,
        text: str,
        document_id: str,
        version_id: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]: ...
