"""Retrieval must survive a restart, even without chromadb installed."""
from __future__ import annotations

from app.config import Settings
from app.services import Services
from tests.conftest import FakeEmbedder


def build_services(tmp_path) -> Services:
    settings = Settings(
        chroma_path=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
        processed_dir=tmp_path / "processed",
        min_chunk_chars=80,
        max_chunk_chars=400,
    )
    for d in (settings.chroma_path, settings.upload_dir, settings.processed_dir):
        d.mkdir(parents=True, exist_ok=True)
    services = Services(settings)
    services.embedder = FakeEmbedder()
    services.chunker.embedder = services.embedder
    return services


TEXT = (
    "1. Data Classification\n"
    "Northwind classifies data into four tiers: Public, Internal, Confidential and Restricted. "
    "Restricted data includes customer financial records and authentication secrets.\n\n"
    "2. Access Control\n"
    "Multi-factor authentication is mandatory for all remote access to production systems. "
    "Least privilege access is reviewed every quarter by team leads.\n"
)


def test_indexes_rebuild_after_restart(tmp_path):
    first = build_services(tmp_path)
    doc = first.ingest_file(TEXT.encode(), "policy.txt")
    assert doc["chunks"] > 0
    assert first.lexical.search("data classification", k=5)

    # Simulate a process restart: new in-memory indexes, same data directory.
    second = build_services(tmp_path)
    assert second.registry.get(doc["document_id"]) is not None
    second.rebuild_indexes()

    lexical = second.lexical.search("data classification", k=5)
    assert lexical, "BM25 index was not rebuilt from persisted chunks"
    vector = second.vector_store.search(
        second.embedder.embed_query("data classification"), k=5
    )
    assert vector, "vector index was not rebuilt from persisted chunks"


def test_delete_removes_persisted_chunks(tmp_path):
    services = build_services(tmp_path)
    doc = services.ingest_file(TEXT.encode(), "policy.txt")
    assert services.registry.load_chunks(doc["document_id"])
    assert services.delete_document(doc["document_id"])
    assert services.registry.load_chunks(doc["document_id"]) == []
