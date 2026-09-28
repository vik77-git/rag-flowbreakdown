from __future__ import annotations

from pathlib import Path

from ..utils.errors import RagError
from .base import DocumentPage, ParsedDocument


class TxtParser:
    extensions = (".txt", ".md")

    def parse(self, path: Path, name: str) -> ParsedDocument:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RagError("This text file could not be read.", 422, detail=str(exc))
        return ParsedDocument(name=name, pages=[DocumentPage(page_number=1, text=text)])
