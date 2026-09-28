from __future__ import annotations

from app.chunking.semantic import Chunk
from app.retrieval.base import Candidate
from app.retrieval.bm25 import Bm25Retriever
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.fusion import reciprocal_rank_fusion


def make_chunks() -> list[Chunk]:
    rows = [
        ("c1", "Multi-factor authentication is mandatory for all remote access.", "doc_a", 1),
        ("c2", "Incident responders acknowledge severity one incidents within fifteen minutes.", "doc_a", 2),
        ("c3", "Least privilege access is reviewed every quarter by team leads.", "doc_b", 1),
    ]
    return [
        Chunk(chunk_id=cid, text=text, metadata={"document_id": doc, "document_name": f"{doc}.txt", "page": page, "position": i})
        for i, (cid, text, doc, page) in enumerate(rows)
    ]


def test_vector_store_search_and_delete(tmp_path, embedder):
    store = ChromaVectorStore(tmp_path / "chroma")
    chunks = make_chunks()
    store.add_documents(chunks, embedder.embed_documents([c.text for c in chunks]))

    hits = store.search(embedder.embed_query("multi-factor authentication remote access"), k=3)
    assert hits and hits[0].chunk_id == "c1"
    assert hits[0].vector_rank == 1

    scoped = store.search(embedder.embed_query("access"), k=3, document_ids=["doc_b"])
    assert {h.metadata["document_id"] for h in scoped} == {"doc_b"}

    assert len(store.get_document_chunks("doc_a")) == 2
    store.delete_document("doc_a")
    assert store.get_document_chunks("doc_a") == []


def test_bm25_exact_keyword_match():
    bm25 = Bm25Retriever()
    bm25.add_documents(make_chunks())
    hits = bm25.search("severity one incidents", k=3)
    assert hits[0].chunk_id == "c2"
    assert hits[0].bm25_rank == 1
    bm25.delete_document("doc_a")
    assert all(h.metadata["document_id"] == "doc_b" for h in bm25.search("access", k=3))


def test_fusion_merges_and_dedupes():
    vector = [
        Candidate(chunk_id="c1", text="alpha text", vector_rank=1, vector_score=0.9),
        Candidate(chunk_id="c2", text="beta text", vector_rank=2, vector_score=0.6),
    ]
    lexical = [
        Candidate(chunk_id="c2", text="beta text", bm25_rank=1, bm25_score=1.0),
        Candidate(chunk_id="c3", text="beta text", bm25_rank=2, bm25_score=0.4),
    ]
    fused = reciprocal_rank_fusion(vector, lexical)
    ids = [c.chunk_id for c in fused]
    assert "c3" not in ids  # duplicate text removed
    c2 = next(c for c in fused if c.chunk_id == "c2")
    assert c2.vector_rank == 2 and c2.bm25_rank == 1
    assert fused == sorted(fused, key=lambda c: c.fusion_score, reverse=True)
