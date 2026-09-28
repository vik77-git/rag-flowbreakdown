# RAG Document Analysis Workspace — Project Summary

A browser-only Retrieval-Augmented Generation workspace. You load a document,
the app parses, chunks and indexes it entirely on the client, then runs a
hybrid retrieval pipeline and asks a Groq-hosted LLM to answer using only the
retrieved passages, with page-level citations.

---

## 1. Stack

| Layer | Choice |
| --- | --- |
| Build tool | Vite 8 (`vite.config.js`, root `web/`, output `dist/`) |
| Language | Vanilla ES modules (JavaScript), no framework |
| UI | Hand-written HTML + CSS (`web/index.html`, `web/src/styles.css`) |
| PDF parsing | `pdfjs-dist` (worker loaded as a bundled asset) |
| DOCX parsing | `mammoth` |
| Retrieval | Custom in-browser BM25 + TF-IDF cosine + RRF (`web/src/rag/retrieve.js`) |
| LLM | Groq OpenAI-compatible API (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`) |
| Persistence | `localStorage` (chat memory only) |
| Hosting | Static deploy; `vercel.json` sets build command, output dir, SPA rewrite |

There is no backend, no database and no server runtime. The legacy Python
FastAPI implementation remains in `rag-app/` as reference only; it is not part
of the deployed app.

## 2. Tools and files

```
web/index.html            app shell: sidebar, workflow strip, chat, evidence rail, metrics
web/src/main.js           UI controller: state, rendering, chat transcript, metrics, memory
web/src/styles.css        design system (Cloud White, focused sidebar)
web/src/rag/parse.js      TXT / MD / PDF / DOCX → normalized pages
web/src/rag/chunk.js      structure-aware chunking + question suggestions
web/src/rag/retrieve.js   HybridIndex: BM25, TF-IDF vector search, RRF fusion
web/src/rag/groq.js       Groq client: key rotation, cooldown, streaming SSE
web/src/rag/pipeline.js   orchestration, intent analysis, reranking, prompts
web/src/assets/security_policy.txt   bundled sample document
.env.example              VITE_ variables to define locally / on Vercel
```

## 3. Type of RAG

**Hybrid-retrieval, reranked, single-corpus RAG with conversational memory.**

- **Hybrid** — lexical BM25 (Okapi, k1 = 1.5, b = 0.75) runs alongside a
  TF-IDF cosine "vector" search over the chunk vocabulary. No embedding model
  or vector database is used; the similarity space is built in the browser.
- **Fusion** — results merge via Reciprocal Rank Fusion (k = 60), so a chunk
  that ranks well in either channel survives.
- **Reranked** — the top 20 fused candidates are scored 0–1 by the fast Groq
  model returning JSON. If no API key is present, the shortlist is too small,
  or the model call fails, a deterministic keyword-overlap heuristic takes over.
- **Adaptive** — a query classifier picks intent (greeting, summarization,
  comparison, explanation, fact extraction), retrieval depth (0–10), answer
  style, and model (20B for simple queries, 120B for summaries and multi-hop).
  Summarization bypasses ranking and takes an even spread across the document.
- **Grounded** — the generator is constrained by non-negotiable rules: answer
  only from context, cite every claim as `[n]`, never invent a document name,
  page or section, and return a fixed "insufficient evidence" line when the
  context does not support an answer.
- **Conversational** — the last six turns are replayed to the model and the
  last twelve are persisted in `localStorage`; "Clear chat + memory" wipes both.

## 4. Workflow

```
Document → parse → chunk → index      (once, on upload or at startup)
Question → analyze → vector + BM25 → RRF → rerank → context → generate → cite
```

1. **Parse** — PDF via pdf.js (text items reassembled into lines), DOCX via
   mammoth, TXT/MD read directly; whitespace normalized, pages preserved.
2. **Chunk** — headings are detected (numbered, markdown, or ALL-CAPS lines);
   text is split into sentences and grown to 200–1600 characters with one
   sentence of overlap. Each chunk keeps document name, page, section and an
   approximate token count.
3. **Index** — term frequencies, document frequencies and lengths are computed
   once for BM25 and TF-IDF.
4. **Analyze** — intent, retrieval depth, multi-hop flag and model selection.
5. **Retrieve** — 30 candidates per channel, fused to at most 40.
6. **Rerank** — LLM JSON scoring or heuristic overlap, top 8 kept.
7. **Build context** — chunks added until the depth limit, a 0.2 relevance
   floor, or a 6000-token budget is hit; each is labelled `Source [n]`.
8. **Generate** — streamed token-by-token from Groq into the transcript.
9. **Report** — citations, ranked evidence, token usage and per-stage latency
   are rendered live; the workflow strip shows each stage as it completes.

## 5. Deliverables

- Single-page static web app, deployable to any static host (Vercel config
  included) with no server component.
- Full client-side ingestion for TXT, MD, PDF and DOCX, plus a bundled sample
  document so the app is usable on first load.
- Hybrid retrieval engine (BM25 + TF-IDF + RRF) and a two-tier reranker with
  an offline-safe fallback.
- Chat interface with streaming answers, greeting and summary handling,
  markdown rendering, citation chips, persistent memory and a clear control.
- Transparency surfaces: live pipeline stages, ranked evidence inspector,
  document preview by page and section, per-query and per-session token and
  latency metrics.
- Configuration and docs: `.env.example`, `vercel.json`, this summary.

## 6. Configuration

Create `.env` from `.env.example`:

```
VITE_GROQ_API_KEY=...        # required for generated answers
VITE_GROQ_API_KEY_2=         # optional, enables round-robin + rate-limit cooldown
VITE_GROQ_BASE_URL=https://api.groq.com/openai/v1
VITE_GROQ_MODEL_20B=openai/gpt-oss-20b
VITE_GROQ_MODEL_120B=openai/gpt-oss-120b
```

```sh
npm install
npm run dev      # http://localhost:8080
npm run build    # → dist/
```

On Vercel, set the same `VITE_` variables under Project Settings →
Environment Variables and redeploy.

**Security note:** `VITE_` variables are inlined into the JavaScript bundle at
build time, so the Groq key is visible to anyone who loads the published page.
This is acceptable for a demo or an internal deployment with a throwaway,
rate-limited key. For a public production deployment the key must move behind
a server proxy.

## 7. Known limits

- Retrieval similarity is lexical (TF-IDF), not semantic embeddings, so
  paraphrased questions with no shared vocabulary retrieve less well.
- One document is active at a time; the index is rebuilt on each upload and
  nothing is persisted between sessions except chat memory.
- Scanned PDFs without a text layer yield no extractable text (no OCR).
