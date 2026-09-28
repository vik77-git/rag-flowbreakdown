"""Batched cross-model reranking: one 20B request scores the whole candidate set."""
from __future__ import annotations

import json
import re

from ..retrieval.base import Candidate
from ..utils.errors import RagError
from ..utils.logging import log_event
from ..utils.tokens import count_tokens
from .heuristic import heuristic_scores

_JSON = re.compile(r"\{.*\}", re.S)

INSTRUCTIONS = (
    "You score how well each candidate passage answers the query.\n"
    "Return ONLY JSON: {\"results\":[{\"chunk_id\":\"C01\",\"score\":0.94}, ...]}\n"
    "Score every candidate between 0.0 (irrelevant) and 1.0 (directly answers the query)."
)


class GroqReranker:
    """Reranker protocol implementation. Falls back to heuristics on bad output."""

    def __init__(self, client, model: str, max_chars_per_candidate: int = 900):
        self.client = client
        self.model = model
        self.max_chars = max_chars_per_candidate

    def _prompt(self, query: str, candidates: list[Candidate]) -> tuple[str, dict[str, Candidate]]:
        labels: dict[str, Candidate] = {}
        parts = []
        for i, cand in enumerate(candidates, start=1):
            label = f"C{i:02d}"
            labels[label] = cand
            parts.append(f"[{label}] {cand.text[: self.max_chars]}")
        return f"Query: {query}\n\nCandidates:\n" + "\n\n".join(parts), labels

    async def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> tuple[list[Candidate], dict]:
        if not candidates:
            return [], {"input_tokens": 0, "output_tokens": 0, "mode": "skipped"}

        prompt, labels = self._prompt(query, candidates)
        messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt}]
        usage = {
            "input_tokens": count_tokens(INSTRUCTIONS + prompt),
            "output_tokens": 0,
            "mode": "llm",
            "estimated": True,
        }

        try:
            content, provider_usage = await self.client.complete(
                messages, self.model, temperature=0, response_format={"type": "json_object"}
            )
            scores = self._parse(content, labels)
            if provider_usage:
                usage = {
                    "input_tokens": provider_usage.get("prompt_tokens", usage["input_tokens"]),
                    "output_tokens": provider_usage.get("completion_tokens", count_tokens(content)),
                    "mode": "llm",
                    "estimated": False,
                }
            if not scores:
                raise ValueError("no usable scores in reranker response")
        except (RagError, ValueError) as exc:
            log_event("rerank_fallback", reason=type(exc).__name__)
            scores = heuristic_scores(query, candidates)
            usage = {"input_tokens": 0, "output_tokens": 0, "mode": "heuristic", "estimated": True}

        for cand in candidates:
            cand.rerank_score = round(float(scores.get(cand.chunk_id, 0.0)), 4)
        ordered = sorted(candidates, key=lambda c: (c.rerank_score or 0.0, c.fusion_score), reverse=True)
        return ordered[:top_n], usage

    @staticmethod
    def _parse(content: str, labels: dict[str, Candidate]) -> dict[str, float]:
        match = _JSON.search(content or "")
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return {}
        raw = data.get("results") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return {}
        scores: dict[str, float] = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            cand = labels.get(str(row.get("chunk_id", "")).strip())
            try:
                score = float(row.get("score"))
            except (TypeError, ValueError):
                continue
            if cand is not None:
                scores[cand.chunk_id] = min(1.0, max(0.0, score if score <= 1 else score / 100.0))
        return scores
