from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Dict, Optional

from app.models.document import ParsedDocument


class BaseDocumentParser(ABC):
    @abstractmethod
    async def parse(
        self,
        file: BinaryIO,
        mime_type: str,
        filename: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument: ...

    @abstractmethod
    def supports(self, mime_type: str) -> bool: ...
