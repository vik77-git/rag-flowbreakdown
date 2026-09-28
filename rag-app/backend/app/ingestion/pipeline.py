"""Upload -> parse -> clean -> structure -> chunk -> metadata (no embedding here)."""
from __future__ import annotations

from pathlib import Path

from ..chunking.semantic import Chunk, SemanticChunker
from ..utils.errors import RagError
from .base import DocumentParser, ParsedDocument
from .cleaner import clean_pages
from .docx_parser import DocxParser
from .pdf_parser import PdfParser
from .structure import build_sections
from .txt_parser import TxtParser

PARSERS: list[DocumentParser] = [PdfParser(), DocxParser(), TxtParser()]
SUPPORTED_EXTENSIONS = tuple(ext for p in PARSERS for ext in p.extensions)


def parser_for(extension: str) -> DocumentParser:
    ext = extension.lower()
    for parser in PARSERS:
        if ext in parser.extensions:
            return parser
    raise RagError(f"Unsupported file type '{ext}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}.", 415)


def parse_document(path: Path, name: str) -> ParsedDocument:
    doc = parser_for(path.suffix).parse(path, name)
    doc.pages = clean_pages(doc.pages)
    if not doc.text.strip():
        raise RagError("No readable text was found in this document.", 422)
    return doc


def chunk_document(doc: ParsedDocument, document_id: str, chunker: SemanticChunker) -> list[Chunk]:
    sections = build_sections(doc)
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(chunker.chunk_section(section, document_id, doc.name, start_position=len(chunks)))
    if not chunks:
        raise RagError("This document produced no usable text chunks.", 422)
    return chunks
