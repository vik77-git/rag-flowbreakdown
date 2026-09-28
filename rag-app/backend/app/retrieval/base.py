from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Candidate:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    vector_score: float | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None


@runtime_checkable
class VectorStore(Protocol):
    def add_documents(self, chunks, embeddings) -> None: ...

    def search(self, embedding, k, document_ids=None) -> list[Candidate]: ...

    def delete_document(self, document_id: str) -> None: ...

    def get_document_chunks(self, document_id: str) -> list[Candidate]: ...


@runtime_checkable
class LexicalRetriever(Protocol):
    def add_documents(self, chunks) -> None: ...

    def search(self, query, k, document_ids=None) -> list[Candidate]: ...

    def delete_document(self, document_id: str) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[Candidate], top_n: int): ...
