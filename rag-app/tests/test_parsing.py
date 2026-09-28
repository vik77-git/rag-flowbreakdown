from __future__ import annotations

import pytest

from app.ingestion.cleaner import clean_pages, clean_text
from app.ingestion.base import DocumentPage
from app.ingestion.pipeline import parse_document, parser_for
from app.ingestion.structure import build_sections, heading_level
from app.ingestion.txt_parser import TxtParser
from app.utils.errors import RagError


def test_txt_parser_reads_text(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Hello world.\n\nSecond paragraph.")
    doc = parse_document(p, "notes.txt")
    assert "Second paragraph" in doc.text
    assert doc.pages[0].page_number == 1


def test_parser_selection_and_rejection():
    assert isinstance(parser_for(".txt"), TxtParser)
    with pytest.raises(RagError):
        parser_for(".exe")


def test_empty_document_rejected(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("   \n\n ")
    with pytest.raises(RagError):
        parse_document(p, "blank.txt")


def test_cleaner_fixes_wraps_and_page_numbers():
    out = clean_text("Multi factor authen-\ntication is\nrequired for admins.\n12\n")
    assert "authentication" in out
    assert "12" not in out.split("\n")


def test_repeated_headers_removed():
    pages = [DocumentPage(page_number=i, text=f"NORTHWIND CONFIDENTIAL\nBody {i} text here.\nFooter line") for i in range(1, 6)]
    cleaned = clean_pages(pages)
    assert all("NORTHWIND CONFIDENTIAL" not in p.text for p in cleaned)
    assert "Body 3 text here." in cleaned[2].text


def test_heading_detection():
    assert heading_level("4.2 Authentication") == 2
    assert heading_level("SECURITY POLICY") == 1
    assert heading_level("This is an ordinary sentence about authentication requirements.") == 0


def test_sections_preserve_hierarchy(tmp_path, sample_text):
    p = tmp_path / "policy.txt"
    p.write_text(sample_text)
    doc = parse_document(p, "policy.txt")
    sections = build_sections(doc)
    titles = [s.title for s in sections]
    assert any(t.startswith("4.2") for t in titles)
    child = next(s for s in sections if s.title.startswith("4.2"))
    assert child.parent.startswith("4")


def test_docx_parser_roundtrip(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "handbook.docx"
    d = docx.Document()
    d.add_heading("1. Secrets Handling", level=1)
    d.add_paragraph("Secrets are read from environment variables.")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Control"
    table.cell(0, 1).text = "Owner"
    table.cell(1, 0).text = "MFA"
    table.cell(1, 1).text = "Security"
    d.save(str(path))

    doc = parse_document(path, "handbook.docx")
    assert "environment variables" in doc.text
    assert any(b.block_type == "table" for b in doc.pages[0].blocks)
    assert any(b.block_type == "heading" for b in doc.pages[0].blocks)
