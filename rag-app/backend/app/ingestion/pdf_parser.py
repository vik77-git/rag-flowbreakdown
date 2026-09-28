from __future__ import annotations

from pathlib import Path

from ..utils.errors import RagError
from .base import DocumentPage, ParsedDocument


class PdfParser:
    extensions = (".pdf",)

    def parse(self, path: Path, name: str) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RagError("PDF support is not installed on the server.", 500, detail=str(exc))
        try:
            reader = PdfReader(str(path))
            pages = [
                DocumentPage(page_number=i + 1, text=(p.extract_text() or ""))
                for i, p in enumerate(reader.pages)
            ]
        except Exception as exc:
            raise RagError("This PDF could not be read. It may be corrupted or image-only.", 422, detail=str(exc))
        return ParsedDocument(name=name, pages=pages)
