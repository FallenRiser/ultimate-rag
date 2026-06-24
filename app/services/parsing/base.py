from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Dict, List, Optional

from pydantic import BaseModel


class PageContent(BaseModel):
    page_no: int
    text: str
    images: List[str] = []   # base64-encoded image strings
    tables: List[str] = []   # markdown-formatted table strings


class ParsedDocument(BaseModel):
    text: str                            # full concatenated text
    pages: List[PageContent] = []
    metadata: Dict[str, Any] = {}


class BaseDocumentParser(ABC):
    @abstractmethod
    async def parse(self, file: BinaryIO, mime_type: str, filename: str) -> ParsedDocument: ...

    @abstractmethod
    def supports(self, mime_type: str) -> bool: ...
