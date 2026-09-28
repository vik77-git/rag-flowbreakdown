"""Heading detection -> Document > Page > Section > Paragraph/Table hierarchy."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .base import Block, DocumentPage, ParsedDocument

_NUMBERED = re.compile(r"^\s*(\d+(\.\d+){0,3})[.)]?\s+(\S.{0,90})$")
_ALLCAPS = re.compile(r"^[A-Z0-9][A-Z0-9 \-/&,'()]{3,70}$")


def heading_level(line: str) -> int:
    """0 = not a heading, otherwise 1..4."""
    s = line.strip()
    if not s or len(s) > 110 or s.endswith((".", ";", ",")):
        return 0
    m = _NUMBERED.match(s)
    if m:
        return min(4, m.group(1).count(".") + 1)
    if _ALLCAPS.match(s):
        return 1
    if s.startswith("#"):
        return min(4, len(s) - len(s.lstrip("#")))
    if len(s.split()) <= 9 and s[0].isupper() and not s.endswith(":"):
        return 2
    return 0


@dataclass
class Section:
    title: str
    parent: str
    page: int
    level: int
    blocks: list[Block] = field(default_factory=list)


def _blocks_for_page(page: DocumentPage) -> list[Block]:
    if page.blocks:
        return page.blocks
    out: list[Block] = []
    for para in [p.strip() for p in page.text.split("\n\n") if p.strip()]:
        lines = para.split("\n")
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                out.append(Block(text="\n".join(buffer).strip(), block_type="paragraph", level=0))
                buffer.clear()

        for line in lines:
            stripped = line.strip()
            standalone = len(lines) == 1
            explicit = bool(_NUMBERED.match(stripped)) or stripped.startswith("#") or bool(_ALLCAPS.match(stripped))
            lvl = heading_level(line) if (standalone or explicit) else 0
            # Inside a paragraph, only an explicit heading marker breaks the text apart.
            if lvl and (standalone or explicit):
                flush()
                out.append(Block(text=stripped, block_type="heading", level=lvl))
            else:
                buffer.append(line)
        flush()
    return out


def build_sections(doc: ParsedDocument) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current: Section | None = None

    for page in doc.pages:
        for block in _blocks_for_page(page):
            lvl = block.level if block.block_type == "heading" else 0
            if lvl:
                title = block.text.strip().lstrip("#").strip()
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                parent = stack[-1][1] if stack else doc.name
                stack.append((lvl, title))
                current = Section(title=title, parent=parent, page=page.page_number, level=lvl)
                sections.append(current)
                continue
            if current is None:
                current = Section(title="Introduction", parent=doc.name, page=page.page_number, level=1)
                sections.append(current)
            current.blocks.append(block)
    return [s for s in sections if s.blocks]
