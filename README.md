# RAG Document Analysis Workspace

A browser-only Retrieval-Augmented Generation workspace. Load a PDF, DOCX, TXT
or Markdown file and the app parses, chunks and indexes it in the browser, runs
hybrid retrieval (BM25 + TF-IDF with reciprocal rank fusion), reranks the
candidates, and streams a cited answer from a Groq-hosted model.

No backend, no database — the whole pipeline runs client-side and deploys as a
static site.

## Quick start

```sh
npm install
cp .env.example .env    # add your VITE_GROQ_API_KEY
npm run dev             # http://localhost:8080
```

Build for production with `npm run build` (output in `dist/`). On Vercel, add
the same `VITE_` variables under Project Settings → Environment Variables.

Without an API key the app still loads, parses documents and shows retrieval
and evidence — only answer generation is disabled.

## Documentation

[`DOCUMENTATION.md`](./DOCUMENTATION.md) covers the stack, tools, RAG type,
full workflow, deliverables and known limits.

## Legacy

The original Python FastAPI implementation is kept for reference in
[`rag-app/`](./rag-app). It is not part of the deployed application.
