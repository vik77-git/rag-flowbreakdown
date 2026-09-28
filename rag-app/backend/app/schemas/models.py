from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    document_ids: list[str] | None = None
    model: str | None = None
    system_prompt: str | None = Field(default=None, max_length=4000)


class Citation(BaseModel):
    index: int
    document: str
    page: int
    section: str
    chunk_id: str


class RetrievalStats(BaseModel):
    vector_candidates: int
    bm25_candidates: int
    hybrid_candidates: int
    reranked: int
    selected_context: int


class Usage(BaseModel):
    query_tokens: int
    context_tokens: int
    reranker_input_tokens: int
    reranker_output_tokens: int
    generator_input_tokens: int
    generator_output_tokens: int
    total_tokens: int
    generator_usage_reported: bool
    reranker_mode: str


class Latency(BaseModel):
    analyze_ms: float
    retrieval_ms: float
    rerank_ms: float
    context_ms: float
    llm_ms: float
    total_ms: float


class RetrievedChunk(BaseModel):
    chunk_id: str
    document: str
    page: int
    section: str
    parent_section: str | None = None
    chunk_type: str | None = None
    token_count: int = 0
    vector_score: float | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None
    selected: bool = False
    text: str


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    model: str
    analysis: dict
    citations: list[Citation]
    chunks: list[RetrievedChunk]
    retrieval: RetrievalStats
    usage: Usage
    latency: Latency


class DocumentSummary(BaseModel):
    document_id: str
    name: str
    extension: str
    size_bytes: int
    pages: int
    chunks: int
    tokens: int
    is_sample: bool = False
    suggestions: list[str] = Field(default_factory=list)
    uploaded_at: float


class SessionUsage(BaseModel):
    queries: int
    total_tokens: int
    avg_tokens_per_query: float
    total_input_tokens: int
    total_output_tokens: int
    documents: int
