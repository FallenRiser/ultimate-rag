from typing import BinaryIO

import httpx

from app.services.parsing.base import BaseDocumentParser, ParsedDocument


class DoclingParser(BaseDocumentParser):
    """Client for a running docling-serve instance.
    POSTs the file to /v1/convert/file and reads back the converted markdown."""

    def __init__(self, url: str, do_ocr: bool = True, do_table_structure: bool = True):
        self.url = url.rstrip("/")
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure

    def supports(self, mime_type: str) -> bool:
        return True  # docling handles all document types

    async def parse(self, file: BinaryIO, mime_type: str, filename: str) -> ParsedDocument:
        file_bytes = file.read()
        # docling-serve accepts conversion options as multipart form fields.
        form = {
            "to_formats": "md",
            "do_ocr": str(self.do_ocr).lower(),
            "do_table_structure": str(self.do_table_structure).lower(),
            "image_export_mode": "placeholder",  # keep base64 image blobs out of chunk text
        }
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{self.url}/v1/convert/file",
                data=form,
                files={"files": (filename, file_bytes, mime_type)},
            )
            response.raise_for_status()
            body = response.json()

        document = body.get("document", body)
        text = document.get("md_content") or document.get("text_content") or ""
        return ParsedDocument(
            text=text,
            metadata={"filename": filename, "mime_type": mime_type, "parser": "docling"},
        )
