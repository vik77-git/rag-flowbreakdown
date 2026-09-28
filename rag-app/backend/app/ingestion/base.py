"""Normalized document representation + parser protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class Block:
    text: str
    block_type: str = "paragraph"  # paragraph | heading | table | list
    level: int = 0


@dataclass
class DocumentPage:
    page_number: int
    text: str = ""
    blocks: list[Block] = field(default_factory=list)


@dataclass
class ParsedDocument:
    name: str
    pages: list[DocumentPage] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


@runtime_checkable
class DocumentParser(Protocol):
    extensions: tuple[str, ...]

    def parse(self, path: Path, name: str) -> ParsedDocument: ...
