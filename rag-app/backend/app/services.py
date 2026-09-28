"""Service container: models are loaded once and shared across requests."""
from __future__ import annotations

import time
import re
from pathlib import Path

from .chunking.semantic import SemanticChunker
from .config import Settings, settings as default_settings
from .embeddings.local import LocalEmbeddingProvider
from .ingestion.pipeline import SUPPORTED_EXTENSIONS, chunk_document, parse_document
from .llm.groq_client import GroqClient
from .reranking.groq_reranker import GroqReranker
from .retrieval.bm25 import Bm25Retriever
from .retrieval.chroma_store import ChromaVectorStore
from .storage.registry import DocumentRegistry
from .utils.logging import log_event


class SessionStore:
    """Per-browser-session usage + query history (in memory for the app session)."""

    def __init__(self) -> None:
        self.usage: dict[str, dict] = {}
        self.queries: dict[str, dict] = {}

    def record(self, session_id: str, usage: dict) -> dict:
        s = self.usage.setdefault(
            session_id, {"queries": 0, "total_tokens": 0, "total_input_tokens": 0, "total_output_tokens": 0}
        )
        s["queries"] += 1
        s["total_tokens"] += usage.get("total_tokens", 0)
        s["total_input_tokens"] += usage.get("generator_input_tokens", 0) + usage.get("reranker_input_tokens", 0)
        s["total_output_tokens"] += usage.get("generator_output_tokens", 0) + usage.get("reranker_output_tokens", 0)
        return s

    def get(self, session_id: str) -> dict:
        return self.usage.get(
            session_id, {"queries": 0, "total_tokens": 0, "total_input_tokens": 0, "total_output_tokens": 0}
        )


class Services:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.embedder = LocalEmbeddingProvider(self.settings.embedding_model)
        self.vector_store = ChromaVectorStore(self.settings.chroma_path)
        self.lexical = Bm25Retriever()
        self.registry = DocumentRegistry(self.settings.processed_dir)
        self.groq = GroqClient(self.settings)
        self.reranker = GroqReranker(self.groq, self.settings.reranker_model)
        self.chunker = SemanticChunker(
            self.embedder,
            threshold_percentile=self.settings.threshold_percentile,
            min_chars=self.settings.min_chunk_chars,
            max_chars=self.settings.max_chunk_chars,
            overlap_sentences=self.settings.overlap_sentences,
        )
        self.sessions = SessionStore()

    # -- lifecycle ---------------------------------------------------------
    async def startup(self) -> None:
        await self.groq.startup()
        self.rebuild_indexes()
        self.load_samples()
        log_event(
            "startup",
            embedding_model=self.settings.embedding_model,
            neural_embeddings=self.embedder.is_neural,
            documents=len(self.registry.all()),
            groq_keys=len(self.settings.groq_keys),
        )

    async def shutdown(self) -> None:
        await self.groq.shutdown()

    def rebuild_indexes(self) -> None:
        """Rebuild BM25 (always in memory) and the vector store from persisted chunks.

        Chunks are persisted by the registry, so retrieval survives a restart even
        when the vector store itself is the in-memory fallback (no chromadb installed).
        """
        from .chunking.semantic import Chunk

        for doc in self.registry.all():
            document_id = doc["document_id"]
            rows = self.registry.load_chunks(document_id)
            if not rows:
                # Legacy documents: recover whatever the vector store still holds.
                rows = [
                    {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
                    for c in self.vector_store.get_document_chunks(document_id)
                ]
                if rows:
                    self.registry.save_chunks(document_id, rows)
            if not rows:
                # Nothing recoverable: re-ingest from the stored upload, else drop
                # the entry so the UI never lists an unsearchable document.
                stored = self.settings.upload_dir / doc.get("stored_name", "")
                if stored.exists():
                    try:
                        self.ingest_file(
                            stored.read_bytes(), doc["name"], is_sample=doc.get("is_sample", False)
                        )
                        self.registry.delete(document_id)
                        log_event("document_reindexed", document_id=document_id)
                        continue
                    except Exception as exc:
                        log_event("reindex_failed", document_id=document_id, error=type(exc).__name__)
                self.registry.delete(document_id)
                log_event("index_rebuild_empty", document_id=document_id)
                continue

            chunks = [
                Chunk(chunk_id=r["chunk_id"], text=r["text"], metadata=dict(r.get("metadata") or {}))
                for r in rows
            ]
            self.lexical.add_documents(chunks)
            if not self.vector_store.get_document_chunks(document_id):
                self.vector_store.add_documents(
                    chunks, self.embedder.embed_documents([c.text for c in chunks])
                )
            log_event("index_rebuilt", document_id=document_id, chunks=len(chunks))

    def load_samples(self) -> None:
        sample_dir = Path(__file__).resolve().parents[2] / "sample_data"
        if not sample_dir.exists():
            return
        existing = {d["name"] for d in self.registry.all()}
        # Keep the first-run workspace focused: one useful default document.
        sample_paths = [sample_dir / "security_policy.txt"]
        for path in sample_paths:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS or path.name in existing:
                continue
            try:
                self.ingest_file(path.read_bytes(), path.name, is_sample=True)
            except Exception as exc:  # never block startup on a sample
                log_event("sample_ingest_failed", file=path.name, error=type(exc).__name__)

    # -- ingestion ---------------------------------------------------------
    def ingest_file(self, content: bytes, safe_name: str, *, is_sample: bool = False) -> dict:
        from .storage.files import store_upload

        document_id, path = store_upload(self.settings.upload_dir, safe_name, content)
        parsed = parse_document(path, safe_name)
        chunks = chunk_document(parsed, document_id, self.chunker)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.add_documents(chunks, embeddings)
        self.lexical.add_documents(chunks)
        self.registry.save_text(
            document_id, [{"page": p.page_number, "text": p.text} for p in parsed.pages]
        )
        self.registry.save_chunks(
            document_id,
            [{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks],
        )
        sections = []
        for chunk in chunks:
            section = str(chunk.metadata.get("section") or "").strip()
            if section and section.lower() not in {s.lower() for s in sections}:
                sections.append(section)
        subject = re.sub(r"[_-]+", " ", Path(safe_name).stem).strip()
        suggestions = [f"Summarize the key points in {subject}."]
        for section in sections[:2]:
            suggestions.append(f"What does the document say about {section}?")
        if len(sections) > 1:
            suggestions.append(f"Compare {sections[0]} and {sections[1]}.")
        suggestions = suggestions[:4]
        doc = {
            "document_id": document_id,
            "name": safe_name,
            "stored_name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": len(content),
            "pages": len(parsed.pages),
            "chunks": len(chunks),
            "tokens": sum(c.metadata.get("token_count", 0) for c in chunks),
            "is_sample": is_sample,
            "suggestions": suggestions,
            "uploaded_at": time.time(),
        }
        self.registry.upsert(doc)
        log_event("document_ingested", document_id=document_id, pages=doc["pages"], chunks=doc["chunks"])
        return doc

    def delete_document(self, document_id: str) -> bool:
        doc = self.registry.delete(document_id)
        if not doc:
            return False
        self.vector_store.delete_document(document_id)
        self.lexical.delete_document(document_id)
        stored = self.settings.upload_dir / doc.get("stored_name", "")
        try:
            if stored.exists():
                stored.unlink()
        except OSError:
            pass
        return True


services = Services()
