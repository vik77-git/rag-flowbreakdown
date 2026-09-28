"""Deterministic fallback reranker: fusion rank + keyword overlap + section signal."""
from __future__ import annotations

from ..retrieval.bm25 import tokenize
from ..retrieval.base import Candidate

_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "what", "which", "does", "do", "for", "on"}


def heuristic_scores(query: str, candidates: list[Candidate]) -> dict[str, float]:
    q_terms = {t for t in tokenize(query) if t not in _STOP}
    max_fusion = max((c.fusion_score for c in candidates), default=0.0) or 1.0
    scores: dict[str, float] = {}
    for cand in candidates:
        terms = set(tokenize(cand.text))
        overlap = len(q_terms & terms) / (len(q_terms) or 1)
        section_hit = 0.1 if q_terms & set(tokenize(str(cand.metadata.get("section", "")))) else 0.0
        base = 0.5 * (cand.fusion_score / max_fusion) + 0.4 * overlap + section_hit
        scores[cand.chunk_id] = min(1.0, round(base, 4))
    return scores
