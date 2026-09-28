"""ChromaDB-backed VectorStore, with an in-memory fallback for offline runs."""
from __future__ import annotations

from pathlib import Path

from ..chunking.semantic import Chunk, cosine_distance
from .base import Candidate


class ChromaVectorStore:
    def __init__(self, path: Path, collection: str = "chunks"):
        self._collection = None
        self._mem: dict[str, tuple[str, dict, list[float]]] = {}
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(path))
            self._collection = client.get_or_create_collection(
                collection, metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self._collection = None

    # -- writes ------------------------------------------------------------
    def add_documents(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if self._collection is not None:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
                embeddings=embeddings,
            )
            return
        for chunk, emb in zip(chunks, embeddings):
            self._mem[chunk.chunk_id] = (chunk.text, chunk.metadata, emb)

    def delete_document(self, document_id: str) -> None:
        if self._collection is not None:
            self._collection.delete(where={"document_id": document_id})
            return
        for cid in [k for k, v in self._mem.items() if v[1].get("document_id") == document_id]:
            self._mem.pop(cid, None)

    # -- reads -------------------------------------------------------------
    def search(self, embedding: list[float], k: int, document_ids: list[str] | None = None) -> list[Candidate]:
        if self._collection is not None:
            where = {"document_id": {"$in": document_ids}} if document_ids else None
            res = self._collection.query(query_embeddings=[embedding], n_results=k, where=where)
            out: list[Candidate] = []
            for rank, (cid, text, meta, dist) in enumerate(
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
            ):
                out.append(
                    Candidate(
                        chunk_id=cid,
                        text=text,
                        metadata=dict(meta or {}),
                        vector_score=round(1.0 - float(dist), 4),
                        vector_rank=rank,
                    )
                )
            return out

        scored = [
            (cid, text, meta, 1.0 - cosine_distance(embedding, emb))
            for cid, (text, meta, emb) in self._mem.items()
            if not document_ids or meta.get("document_id") in document_ids
        ]
        scored.sort(key=lambda r: r[3], reverse=True)
        return [
            Candidate(chunk_id=c, text=t, metadata=dict(m), vector_score=round(s, 4), vector_rank=i)
            for i, (c, t, m, s) in enumerate(scored[:k], start=1)
        ]

    def get_document_chunks(self, document_id: str) -> list[Candidate]:
        if self._collection is not None:
            res = self._collection.get(where={"document_id": document_id})
            rows = list(zip(res["ids"], res["documents"], res["metadatas"]))
        else:
            rows = [
                (cid, text, meta)
                for cid, (text, meta, _) in self._mem.items()
                if meta.get("document_id") == document_id
            ]
        candidates = [Candidate(chunk_id=c, text=t, metadata=dict(m or {})) for c, t, m in rows]
        return sorted(candidates, key=lambda c: c.metadata.get("position", 0))
