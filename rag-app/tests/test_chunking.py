from __future__ import annotations

from app.chunking.semantic import percentile, split_sentences
from app.ingestion.base import Block
from app.ingestion.structure import Section


def make_section(text: str, title: str = "4.2 Authentication") -> Section:
    return Section(title=title, parent="4 Authentication", page=14, level=2, blocks=[Block(text=text)])


def test_percentile_helper():
    assert percentile([0.1, 0.2, 0.3, 0.4], 50) == 0.25
    assert percentile([0.5], 95) == 0.5


def test_sentence_split():
    assert len(split_sentences("One sentence here. Another sentence follows! A third?")) == 3


def test_chunks_respect_max_and_min(chunker):
    body = " ".join(f"Authentication requirement number {i} is enforced on every request." for i in range(40))
    chunks = chunker.chunk_section(make_section(body), "doc_1", "policy.txt")
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 + 200 for c in chunks)
    assert all(len(c.text) >= 60 for c in chunks)


def test_metadata_is_preserved(chunker):
    chunks = chunker.chunk_section(make_section("MFA is mandatory for all remote access. " * 8), "doc_1", "policy.txt")
    meta = chunks[0].metadata
    assert meta["document_id"] == "doc_1"
    assert meta["document_name"] == "policy.txt"
    assert meta["section"] == "4.2 Authentication"
    assert meta["parent_section"] == "4 Authentication"
    assert meta["page"] == 14
    assert meta["token_count"] > 0
    assert meta["chunk_type"] == "paragraph"


def test_positions_are_sequential(chunker):
    body = " ".join(f"Control {i} applies to production systems and is reviewed quarterly." for i in range(30))
    chunks = chunker.chunk_section(make_section(body), "doc_1", "policy.txt", start_position=5)
    positions = [c.metadata["position"] for c in chunks]
    assert positions == list(range(5, 5 + len(chunks)))


def test_semantic_boundary_splits_topics(chunker):
    a = "Password length must be at least fourteen characters. Passwords are screened against breach lists. Password reuse is prohibited across systems. "
    b = "Incident responders acknowledge severity one pages within fifteen minutes. Incident channels are declared within thirty minutes. Post incident reviews are blameless. "
    chunks = chunker.chunk_section(make_section(a + b), "doc_1", "policy.txt")
    assert len(chunks) >= 2


def test_tables_become_their_own_chunk(chunker):
    section = Section(
        title="2. Data Classification", parent="Policy", page=3, level=1,
        blocks=[Block(text="Tiers are listed below and reviewed annually by the board." * 3),
                Block(text="Tier | Encryption\nRestricted | AES-256", block_type="table")],
    )
    chunks = chunker.chunk_section(section, "doc_2", "policy.txt")
    types = [c.metadata["chunk_type"] for c in chunks]
    assert "table" in types
