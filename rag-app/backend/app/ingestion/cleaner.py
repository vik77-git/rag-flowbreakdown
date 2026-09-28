"""Conservative text cleaning: fixes extraction artefacts, not meaning."""
from __future__ import annotations

import re
from collections import Counter

from .base import DocumentPage

_PAGE_NUM = re.compile(r"^\s*(page\s+)?\d+\s*(/\s*\d+)?\s*$", re.I)
_WS = re.compile(r"[ \t]{2,}")
_BLANKS = re.compile(r"\n{3,}")
_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")
_SOFT_WRAP = re.compile(r"(?<![.!?:;])\n(?=[a-z(])")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _SOFT_WRAP.sub(" ", text)
    lines = [_WS.sub(" ", ln).rstrip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not _PAGE_NUM.match(ln)]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def strip_repeated_edges(pages: list[DocumentPage]) -> list[DocumentPage]:
    """Drop headers/footers that repeat on most pages."""
    if len(pages) < 3:
        return pages
    firsts, lasts = Counter(), Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.text.split("\n") if ln.strip()]
        if lines:
            firsts[lines[0]] += 1
            lasts[lines[-1]] += 1
    threshold = max(2, int(len(pages) * 0.6))
    repeated = {t for t, c in (*firsts.items(), *lasts.items()) if c >= threshold and len(t) < 120}
    if not repeated:
        return pages
    for p in pages:
        kept = [ln for ln in p.text.split("\n") if ln.strip() not in repeated]
        p.text = _BLANKS.sub("\n\n", "\n".join(kept)).strip()
        p.blocks = [b for b in p.blocks if b.text.strip() not in repeated]
    return pages


def clean_pages(pages: list[DocumentPage]) -> list[DocumentPage]:
    for p in pages:
        p.text = clean_text(p.text)
        for b in p.blocks:
            b.text = clean_text(b.text)
        p.blocks = [b for b in p.blocks if b.text]
    return strip_repeated_edges(pages)
