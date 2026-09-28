# Advanced RAG — Python source project (download & run locally)

Generate the complete FastAPI + ChromaDB + Groq RAG codebase as files in this repo. Nothing runs in the preview here (this environment serves a TypeScript web app on an edge runtime, and Python/Chroma/sentence-transformers cannot execute in it), so the deliverable is a runnable-on-your-machine project you download and start with `uvicorn`. No Lovable Cloud, no secrets stored here — the Groq key goes in your local `.env`.

## Model policy (as you specified)

| Task | Model |
| --- | --- |
| Query analysis | 20B (heuristics first, LLM only when ambiguous) |
| Reranking | 20B — one batched call for all candidates |
| Simple answer | 20B |
| Complex / multi-hop answer | 120B |
| Streaming generation | user-selected or auto |

Reranker sends the whole candidate set in a single structured request and expects
`{"results":[{"chunk_id":"C07","score":0.94}, ...]}`, then validates the JSON, normalizes
scores, sorts, and keeps top 5–8. Malformed output falls back to a heuristic scorer so a
bad response never breaks a query.

## Layout

```text
rag-app/
├── backend/app/
│   ├── main.py            config.py
│   ├── api/               health, documents, query, session routes
│   ├── ingestion/         pdf_parser, docx_parser, txt_parser, cleaner, structure
│   ├── chunking/          semantic chunker (percentile threshold + safety limits)
│   ├── embeddings/        EmbeddingProvider protocol + sentence-transformers impl
│   ├── retrieval/         chroma_store (VectorStore), bm25 (LexicalRetriever), fusion (RRF)
│   ├── reranking/         Reranker protocol + batched Groq 20B reranker + heuristic fallback
│   ├── reasoning/         query analyzer, dynamic depth, context builder, prompt builder
│   ├── llm/               groq_client (streaming, usage), LLMProvider protocol
│   ├── storage/           document registry (JSON), file store, path-traversal guards
│   ├── schemas/           Pydantic request/response models
│   └── utils/             tokens, logging, timing, errors
├── frontend/              index.html, styles.css, app.js (vanilla, dark navy enterprise)
├── sample_data/           security_policy.pdf-source, developer_handbook.docx, architecture_guide.txt
├── tests/                 parsing, chunking, retrieval, reranking, query, security
├── requirements.txt  .env.example  README.md  run.sh
```

## Pipeline

Ingestion: upload → extension + MIME + size + emptiness validation → parser → cleaner
(repeated headers/footers, page numbers, broken wraps) → structure detection (headings →
sections → paragraphs/tables) → semantic chunking → metadata → batched embeddings →
ChromaDB + BM25 index.

Query: analyzer (intent, keywords, multi-hop, depth, answer style) → vector top-30 +
BM25 top-30 → RRF fusion + dedupe → batched 20B rerank → context builder (budget,
dedupe, structure-preserving, relevance floor) → prompt with `[Source n]` blocks →
Groq streaming → citations mapped back to chunk metadata.

Insufficient evidence below the configurable relevance threshold returns the explicit
"could not find sufficient evidence" answer with no citations.

## UI (vanilla JS, SSE-driven)

Dashboard: document list + upload with per-stage progress (Uploading → Parsing →
Analyzing structure → Chunking → Embeddings → Indexing → Ready); document preview pane
(PDF page list, DOCX/TXT text with section tree); question box with the six example query
buttons; model selector (20B / 120B with plain descriptions); retrieval workflow strip
with per-stage status, duration and counts; retrieved-chunks pane (document, page,
section, chunk id, vector/BM25 rank, rerank score, tokens, expand/collapse); streaming
answer with clickable citations; Query Usage panel using your expanded breakdown
(query, retrieval context, reranker input/output, generator input/output, total);
session usage aggregate.

## Endpoints

`GET /`, `GET /api/health`, `POST /api/documents/upload`, `GET /api/documents`,
`GET /api/documents/{id}`, `DELETE /api/documents/{id}`, `POST /api/query`,
`POST /api/query/stream` (SSE), `GET /api/query/{query_id}`, `GET /api/session/usage`.

SSE events: `query_started`, `query_analyzed`, `vector_search_complete`,
`bm25_search_complete`, `hybrid_search_complete`, `reranking_complete`, `context_built`,
`generation_started`, `token`, `generation_complete`, `usage`, `query_complete`, `error`.

## Technical notes

- Token accounting: Groq-reported usage is authoritative per call; app-side counts
  (query, context) come from a tokenizer and are labelled estimates in the API and UI.
- Models load once at startup via lifespan; CPU-heavy embed/rerank work runs in a thread
  pool so the event loop stays free. One shared `httpx.AsyncClient` for Groq.
- Every layer sits behind a Protocol (DocumentParser, EmbeddingProvider, VectorStore,
  LexicalRetriever, Reranker, LLMProvider) so Chroma/embeddings/Groq are swappable.
- Structured JSON logs per request: request_id, query_id, document_ids, model, retrieval
  counts, usage, latencies, error status — no document content.
- Security: env-only keys, sanitized filenames, uploads outside code dirs, traversal
  guards, Pydantic validation, no stack traces or paths in responses.
- Tests are pytest and run offline: embedding and Groq providers are faked, so
  parsing/chunking/fusion/rerank-ordering/citation/security tests pass with no API key.
- I write and review the code but cannot execute Python here, so the test suite is
  authored, not run. First local `pytest` run is your verification step.

## Verify locally

```bash
pip install -r requirements.txt
cp .env.example .env      # add GROQ_API_KEY
uvicorn app.main:app --reload --app-dir backend
pytest
```
