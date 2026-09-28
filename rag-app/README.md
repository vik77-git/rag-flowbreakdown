# RAG using LangChain

A production-shaped Retrieval-Augmented Generation application: FastAPI backend, vanilla
JS dashboard, hybrid retrieval (dense + lexical) with reciprocal rank fusion, an LLM
reranker with a heuristic fallback, streaming answers with inline citations, and full
token/latency accounting per query.

Documents in `sample_data/` are fictional samples written for demos.

---

## 1. Overview

- **Ingest** PDF / DOCX / TXT → clean → detect structure (headings, sections, tables) →
  semantic chunking driven by embedding distance, not fixed character windows.
- **Retrieve** every query through two independent retrievers (Chroma vectors and BM25),
  fused with Reciprocal Rank Fusion so neither modality dominates.
- **Rerank** the fused candidate set in a *single batched* call to the 20B model; if the
  model returns unusable output, a deterministic heuristic reranker takes over and the
  response records `reranker_mode`.
- **Answer** with the Groq chat model, streamed token by token over SSE, with numbered
  citations `[n]` bound to the exact chunks that were sent as context.
- **Account** for tokens and latency at every stage and surface them in the UI.

## 2. Architecture

```text
              ┌────────────────────────── frontend/ (vanilla JS) ──────────────────────────┐
              │ workflow strip · retrieved chunks · streaming answer · usage/latency panes │
              └───────────────▲─────────────────────────────────────────┬──────────────────┘
                              │ SSE events                              │ REST
┌─────────────────────────────┴─────────────────────────────────────────▼──────────────────┐
│ FastAPI (backend/app/main.py)                                                            │
│                                                                                          │
│  api/documents.py   api/query.py   api/session.py   api/health.py                        │
│         │                │                                                               │
│         │                └────────────► reasoning/orchestrator.py ─────────────┐         │
│         │                                     │                                │         │
│         ▼                                     ▼                                ▼         │
│  ingestion/pipeline.py              reasoning/analyzer.py              llm/groq_client.py │
│   parsers → cleaner → structure      (intent, depth, keywords)          llm/key_rotator.py│
│         │                                     │                                          │
│         ▼                                     ▼                                          │
│  chunking/semantic.py               retrieval/chroma_store.py  retrieval/bm25.py          │
│         │                                     └────────► retrieval/fusion.py (RRF)        │
│         ▼                                                        │                        │
│  embeddings/local.py                                             ▼                        │
│   (sentence-transformers)                            reranking/groq_reranker.py           │
│                                                        └─ fallback: reranking/heuristic.py│
│                                                                  │                        │
│                                                                  ▼                        │
│                                                     reasoning/context.py  prompts.py      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

Storage: Chroma persists vectors under `CHROMA_PATH`; uploads land in `UPLOAD_DIR`,
parsed/chunked artefacts in `PROCESSED_DIR`. Logs are single-line JSON on stdout.

## 3. Setup

Requirements: Python 3.10+ and a Groq API key (two keys recommended, see §7).

```bash
cd rag-app
cp .env.example .env         # then paste GROQ_API_KEY (and GROQ_API_KEY_2)
./run.sh                     # installs requirements and starts uvicorn on :8000
```

Then open <http://localhost:8000>. The API docs live at `/docs`.

Manual equivalent:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir backend --port 8000
```

`requirements.txt` is intentionally light (a few MB, no ML frameworks). Optional
upgrades are separate:

```bash
pip install -r requirements-optional.txt   # neural embeddings + Chroma + tiktoken (~2 GB)
pip install -r requirements-dev.txt        # core + pytest + reportlab
```

Without the optional set the app still runs end to end: a deterministic hashing
embedder replaces sentence-transformers, an in-memory store replaces Chroma, and
token counts fall back to a character estimate. With it installed, first start
downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB) once.
Sample documents are ingested automatically at startup, so you can query immediately.

Tests:

```bash
python -m pytest tests -q      # 40 tests, no network and no model download required
```


The suite uses a deterministic hashing embedder, so it runs fully offline.

## 4. How the pipeline works

### 4.1 Parsing and structure
`ingestion/` extracts text per page (`pypdf`, `python-docx`, plain text), repairs hyphenated
line breaks and de-duplicates headers/footers (`cleaner.py`), then `structure.py` assigns each
block to a section using numbered headings (`4.2 Multi-Factor Authentication`), ALL-CAPS
titles, markdown `#` levels, and short title-cased lines. Sections keep a parent pointer so a
chunk can be cited as *document → section → page*.

### 4.2 Semantic chunking
`chunking/semantic.py` splits each section into sentences, embeds them, and measures the
distance between consecutive sentences. A break is placed where the distance exceeds the
`SEMANTIC_THRESHOLD_PERCENTILE` (default 95th) of that section's distances — so chunks end at
topic shifts. `MIN_CHUNK_CHARS` merges undersized fragments forward, `MAX_CHUNK_CHARS` splits
oversized ones, and `CHUNK_OVERLAP_SENTENCES` carries context across boundaries. Tables are
kept whole. Every chunk stores `document_id`, `document_name`, `page`, `section`,
`parent_section`, `position` and `token_count`.

### 4.3 Retrieval and fusion
Each query runs a vector search (`TOP_K_VECTOR`) and a BM25 search (`TOP_K_BM25`) in
parallel. `fusion.py` combines them with RRF: `score = Σ 1 / (k + rank)`, `k = 60`. Rank-based
fusion means a chunk found by both retrievers rises to the top without any score
normalisation between incomparable scales. The UI shows each chunk's vector rank, BM25 rank,
fusion score and rerank score.

### 4.4 Reranking
One request sends all fused candidates (truncated to 900 chars each, labelled `C01…Cnn`) to
the 20B model, which returns `{"results":[{"chunk_id":"C01","score":0.94}, ...]}`. Batching
keeps reranking to a single round trip instead of one call per candidate. Malformed or
incomplete JSON triggers `reranking/heuristic.py` (keyword overlap, phrase hits, title
matches, position prior) and `usage.reranker_mode` becomes `heuristic`.

### 4.5 Context building and token accounting
`reasoning/context.py` applies a relevance floor (`RELEVANCE_THRESHOLD`), drops
near-duplicates, restores the original document order for readability, then fills up to
`CONTEXT_TOKEN_BUDGET` tokens (≈40 tokens reserved per chunk for its citation header).
Tokens are counted with `tiktoken` in `utils/tokens.py`.

The response reports, separately: `query_tokens`, `context_tokens`,
`reranker_input_tokens` / `reranker_output_tokens`, and
`generator_input_tokens` / `generator_output_tokens`. When Groq returns its own usage block
the generator figures are exact and `generator_usage_reported` is `true`; otherwise they are
`tiktoken` estimates. Session totals are aggregated in `/api/session/usage`.

### 4.6 Answer, citations, insufficient evidence
`reasoning/prompts.py` always wraps the effective system prompt — including a custom one
typed in the UI — in master rules: answer only from the context, cite `[n]`, never invent
sources. If nothing clears the relevance floor, the orchestrator streams an explicit
"insufficient evidence" answer instead of guessing, and still reports usage and latency.

## 5. API

Base URL `http://localhost:8000`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service status, model names, document count, key-rotation state |
| `GET` | `/api/documents` | List ingested documents (`DocumentSummary[]`) |
| `POST` | `/api/documents/upload` | Multipart upload (`file`); parses, chunks and indexes |
| `GET` | `/api/documents/{document_id}` | Document preview: metadata, pages, sections, chunks |
| `DELETE` | `/api/documents/{document_id}` | Remove document and its vectors (samples protected) |
| `POST` | `/api/query` | Non-streaming answer (`QueryResponse`) |
| `POST` | `/api/query/stream` | Same pipeline as SSE |
| `GET` | `/api/query/{query_id}` | Replay a completed query from the session cache |
| `GET` | `/api/session/usage` | Aggregate token usage for the session |

Query body:

```json
{
  "query": "How do the MFA rules compare with the access-review policy?",
  "document_ids": ["optional", "subset"],
  "model": "openai/gpt-oss-120b",
  "system_prompt": "optional custom prompt, still screened by master rules"
}
```

SSE event sequence on `/api/query/stream` (`event:` + JSON `data:`):

`query_started` → `query_analyzed` → `vector_search_complete` → `bm25_search_complete` →
`hybrid_search_complete` → `reranking_complete` → `context_built` →
`generation_started` → many `token` → `generation_complete` → `usage` → `query_complete`.
Errors arrive as an `error` event with a human-readable `message`.

`QueryResponse` carries `answer`, `citations[]`, `chunks[]` (every candidate with its scores
and a `selected` flag), `retrieval` counts, `usage` and `latency` breakdowns.

## 6. Configuration

All settings come from `.env` (see `.env.example`). The ones worth tuning:

| Variable | Default | Effect |
| --- | --- | --- |
| `GROQ_MODEL_120B` / `GROQ_MODEL_20B` | `openai/gpt-oss-120b` / `-20b` | Generator / reranker |
| `TOP_K_VECTOR`, `TOP_K_BM25` | 30 | Candidates per retriever before fusion |
| `TOP_K_RERANK` | 8 | Candidates kept after reranking |
| `SEMANTIC_THRESHOLD_PERCENTILE` | 95 | Higher = fewer, larger chunks |
| `MIN_CHUNK_CHARS` / `MAX_CHUNK_CHARS` | 200 / 1800 | Chunk size guardrails |
| `CONTEXT_TOKEN_BUDGET` | 6000 | Max tokens of context per answer |
| `RELEVANCE_THRESHOLD` | 0.25 | Floor below which a chunk is never used |
| `MAX_UPLOAD_MB` | 25 | Upload size limit |

## 7. Key rotation and rate limits

`llm/key_rotator.py` round-robins between `GROQ_API_KEY` and `GROQ_API_KEY_2`. Each key
serves `GROQ_REQUESTS_PER_KEY` (default 5) requests, then enters a
`GROQ_KEY_COOLDOWN_SECONDS` (default 30 s) cooldown while the other key takes over. If both
are cooling down, callers wait rather than fail. One key works fine — you just hit the
cooldown pause more often. Live state is visible in `/api/health`.

## 8. Security notes

- Uploads are extension- and size-checked, stored under a generated ID, and never executed;
  filenames are sanitised before touching disk (`storage/files.py`).
- Custom system prompts cannot override the master rules or the citation requirement.
- API keys are read from the environment only and are never returned by any endpoint or log
  line; `/api/health` reports key *state*, not key values.
- There is no authentication layer — this is a single-user local app. Put it behind a proxy
  with auth before exposing it.

## 9. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `No Groq API key is configured` (503) | `GROQ_API_KEY` missing from `.env`; restart after editing |
| Long pause on first start | Embedding model download; subsequent starts are instant |
| `reranker_mode: heuristic` in the usage panel | The 20B model returned unparsable JSON — answers still work, quality dips slightly; retry or switch `RERANKER_MODEL` |
| Answer says the context is insufficient | Nothing cleared `RELEVANCE_THRESHOLD` — lower it, raise `TOP_K_RERANK`, or upload a document that covers the question |
| Groq 429s | Both keys are in cooldown; add `GROQ_API_KEY_2` or lower `GROQ_REQUESTS_PER_KEY` |
| Upload rejected | Unsupported extension or over `MAX_UPLOAD_MB` |
| Stale results after deleting a document | Delete via `DELETE /api/documents/{id}`; deleting files by hand leaves vectors behind |
| Chroma errors after an upgrade | Remove `data/chroma` and restart to rebuild the index |

## 10. Layout

```text
rag-app/
  backend/app/    api/ chunking/ embeddings/ ingestion/ llm/ reasoning/ reranking/
                  retrieval/ storage/ utils/ schemas/ config.py main.py services.py
  frontend/       index.html  app.js  styles.css
  sample_data/    three fictional sample documents
  scripts/        make_sample_binaries.py (generate PDF/DOCX variants)
  tests/          40 pytest tests across parsing, chunking, retrieval, reranking,
                  key rotation, query flow and security
  requirements.txt  .env.example  run.sh
```
