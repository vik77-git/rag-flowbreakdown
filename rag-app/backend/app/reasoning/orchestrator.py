"""RAG orchestrator. Emits pipeline events; the API turns them into SSE or a response."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncIterator

from ..retrieval.base import Candidate
from ..retrieval.fusion import reciprocal_rank_fusion
from ..utils.logging import log_event
from ..utils.timing import stopwatch
from ..utils.tokens import count_tokens
from .analyzer import analyze_query
from .context import build_context, citations_for
from .prompts import INSUFFICIENT_EVIDENCE, build_messages


def chunk_payload(cand: Candidate, selected_ids: set[str]) -> dict:
    meta = cand.metadata
    return {
        "chunk_id": cand.chunk_id,
        "document": meta.get("document_name", "unknown"),
        "page": meta.get("page", 1),
        "section": meta.get("section", "n/a"),
        "parent_section": meta.get("parent_section"),
        "chunk_type": meta.get("chunk_type"),
        "token_count": meta.get("token_count", count_tokens(cand.text)),
        "vector_score": cand.vector_score,
        "vector_rank": cand.vector_rank,
        "bm25_score": cand.bm25_score,
        "bm25_rank": cand.bm25_rank,
        "fusion_score": cand.fusion_score,
        "rerank_score": cand.rerank_score,
        "selected": cand.chunk_id in selected_ids,
        "text": cand.text,
    }


class Orchestrator:
    def __init__(self, services):
        self.s = services

    def pick_model(self, requested: str | None, analysis: dict) -> str:
        cfg = self.s.settings
        if requested == "auto" or requested is None:
            return cfg.model_120b if analysis.get("multi_hop") else cfg.model_20b
        return self.s.groq.validate_model(requested)

    async def run(
        self,
        query: str,
        *,
        document_ids: list[str] | None,
        model: str | None,
        system_prompt: str | None,
        session_id: str,
    ) -> AsyncIterator[dict]:
        cfg = self.s.settings
        query_id = f"q_{uuid.uuid4().hex[:10]}"
        started = time.perf_counter()
        document_ids = [d for d in (document_ids or []) if d] or None
        yield {"event": "query_started", "data": {"query_id": query_id, "query": query}}

        # 1. analysis
        with stopwatch() as ms:
            analysis = analyze_query(query, base_depth=min(cfg.top_k_rerank, 5))
        analyze_ms = ms()
        chosen_model = self.pick_model(model, analysis)
        yield {"event": "query_analyzed", "data": {"analysis": analysis, "model": chosen_model, "duration_ms": analyze_ms}}

        # 2. hybrid retrieval
        loop = asyncio.get_running_loop()
        with stopwatch() as ms:
            embedding = await loop.run_in_executor(None, self.s.embedder.embed_query, query)
            vector = await loop.run_in_executor(
                None, lambda: self.s.vector_store.search(embedding, cfg.top_k_vector, document_ids)
            )
            yield {"event": "vector_search_complete", "data": {"count": len(vector)}}
            lexical = await loop.run_in_executor(
                None, lambda: self.s.lexical.search(query, cfg.top_k_bm25, document_ids)
            )
            yield {"event": "bm25_search_complete", "data": {"count": len(lexical)}}
            fused = reciprocal_rank_fusion(vector, lexical, limit=cfg.top_k_vector + cfg.top_k_bm25)
        retrieval_ms = ms()
        yield {"event": "hybrid_search_complete", "data": {"count": len(fused), "duration_ms": retrieval_ms}}

        if not fused:
            async for evt in self._finish_empty(
                query, query_id, chosen_model, analysis, analyze_ms, retrieval_ms, started, session_id,
                message="No indexed documents matched this question. Upload a document or pick another one.",
            ):
                yield evt
            return

        # 3. rerank (single batched 20B call)
        with stopwatch() as ms:
            top, rerank_usage = await self.s.reranker.rerank(query, fused, cfg.top_k_rerank)
        rerank_ms = ms()
        yield {
            "event": "reranking_complete",
            "data": {"input": len(fused), "output": len(top), "mode": rerank_usage["mode"], "duration_ms": rerank_ms},
        }

        # 4. context
        with stopwatch() as ms:
            selected, context, context_tokens = build_context(
                top,
                depth=analysis["retrieval_depth"],
                token_budget=cfg.context_token_budget,
                relevance_threshold=cfg.relevance_threshold,
            )
        context_ms = ms()
        selected_ids = {c.chunk_id for c in selected}
        chunks = [chunk_payload(c, selected_ids) for c in top]
        yield {
            "event": "context_built",
            "data": {"count": len(selected), "context_tokens": context_tokens, "duration_ms": context_ms, "chunks": chunks},
        }

        if not selected:
            async for evt in self._finish_empty(
                query, query_id, chosen_model, analysis, analyze_ms, retrieval_ms, started, session_id,
                message=INSUFFICIENT_EVIDENCE, rerank_ms=rerank_ms, context_ms=context_ms,
                chunks=chunks, hybrid=len(fused), vector=len(vector), bm25=len(lexical), reranked=len(top),
                rerank_usage=rerank_usage,
            ):
                yield evt
            return

        # 5. generation
        messages = build_messages(query, context, analysis, system_prompt=system_prompt or cfg.system_prompt)
        prompt_text = "".join(m["content"] for m in messages)
        yield {"event": "generation_started", "data": {"model": chosen_model}}

        answer_parts: list[str] = []
        provider_usage: dict = {}
        with stopwatch() as ms:
            async for event in self.s.groq.stream(messages, chosen_model, temperature=0.2):
                if "delta" in event:
                    answer_parts.append(event["delta"])
                    yield {"event": "token", "data": {"text": event["delta"]}}
                elif "usage" in event:
                    provider_usage = event["usage"] or {}
        llm_ms = ms()
        answer = "".join(answer_parts).strip() or INSUFFICIENT_EVIDENCE
        yield {"event": "generation_complete", "data": {"duration_ms": llm_ms}}

        gen_in = provider_usage.get("prompt_tokens") or count_tokens(prompt_text)
        gen_out = provider_usage.get("completion_tokens") or count_tokens(answer)
        usage = {
            "query_tokens": count_tokens(query),
            "context_tokens": context_tokens,
            "reranker_input_tokens": rerank_usage["input_tokens"],
            "reranker_output_tokens": rerank_usage["output_tokens"],
            "generator_input_tokens": gen_in,
            "generator_output_tokens": gen_out,
            "total_tokens": rerank_usage["input_tokens"] + rerank_usage["output_tokens"] + gen_in + gen_out,
            "generator_usage_reported": bool(provider_usage),
            "reranker_mode": rerank_usage["mode"],
        }
        total_ms = round((time.perf_counter() - started) * 1000, 1)
        latency = {
            "analyze_ms": analyze_ms,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "context_ms": context_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
        }
        response = {
            "query_id": query_id,
            "query": query,
            "answer": answer,
            "model": chosen_model,
            "analysis": analysis,
            "citations": citations_for(selected),
            "chunks": chunks,
            "retrieval": {
                "vector_candidates": len(vector),
                "bm25_candidates": len(lexical),
                "hybrid_candidates": len(fused),
                "reranked": len(top),
                "selected_context": len(selected),
            },
            "usage": usage,
            "latency": latency,
        }
        self.s.sessions.queries[query_id] = response
        session = self.s.sessions.record(session_id, usage)
        log_event(
            "query_complete",
            query_id=query_id,
            model=chosen_model,
            document_ids=document_ids,
            retrieval=response["retrieval"],
            usage=usage,
            latency=latency,
        )
        yield {"event": "usage", "data": {"usage": usage, "latency": latency, "session": session}}
        yield {"event": "query_complete", "data": response}

    async def _finish_empty(
        self, query, query_id, model, analysis, analyze_ms, retrieval_ms, started, session_id, *,
        message, rerank_ms=0.0, context_ms=0.0, chunks=None, hybrid=0, vector=0, bm25=0, reranked=0,
        rerank_usage=None,
    ):
        rerank_usage = rerank_usage or {"input_tokens": 0, "output_tokens": 0, "mode": "skipped"}
        usage = {
            "query_tokens": count_tokens(query),
            "context_tokens": 0,
            "reranker_input_tokens": rerank_usage["input_tokens"],
            "reranker_output_tokens": rerank_usage["output_tokens"],
            "generator_input_tokens": 0,
            "generator_output_tokens": 0,
            "total_tokens": rerank_usage["input_tokens"] + rerank_usage["output_tokens"],
            "generator_usage_reported": False,
            "reranker_mode": rerank_usage["mode"],
        }
        total_ms = round((time.perf_counter() - started) * 1000, 1)
        response = {
            "query_id": query_id,
            "query": query,
            "answer": message,
            "model": model,
            "analysis": analysis,
            "citations": [],
            "chunks": chunks or [],
            "retrieval": {
                "vector_candidates": vector,
                "bm25_candidates": bm25,
                "hybrid_candidates": hybrid,
                "reranked": reranked,
                "selected_context": 0,
            },
            "usage": usage,
            "latency": {
                "analyze_ms": analyze_ms,
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "context_ms": context_ms,
                "llm_ms": 0.0,
                "total_ms": total_ms,
            },
        }
        self.s.sessions.queries[query_id] = response
        session = self.s.sessions.record(session_id, usage)
        log_event("query_insufficient_evidence", query_id=query_id, hybrid_candidates=hybrid)
        for text in (message,):
            yield {"event": "token", "data": {"text": text}}
        yield {"event": "usage", "data": {"usage": usage, "latency": response["latency"], "session": session}}
        yield {"event": "query_complete", "data": response}
