"""Reciprocal rank fusion + deduplication of vector and lexical candidates."""
from __future__ import annotations

import hashlib

from .base import Candidate


def _fingerprint(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


def reciprocal_rank_fusion(
    vector: list[Candidate], lexical: list[Candidate], *, k: int = 60, limit: int = 50
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}

    for group in (vector, lexical):
        for cand in group:
            existing = merged.get(cand.chunk_id)
            if existing is None:
                merged[cand.chunk_id] = cand
                existing = cand
            else:
                existing.vector_score = existing.vector_score if cand.vector_score is None else cand.vector_score
                existing.vector_rank = existing.vector_rank or cand.vector_rank
                existing.bm25_score = existing.bm25_score if cand.bm25_score is None else cand.bm25_score
                existing.bm25_rank = existing.bm25_rank or cand.bm25_rank
            rank = cand.vector_rank or cand.bm25_rank or 0
            if rank:
                existing.fusion_score += 1.0 / (k + rank)

    # drop duplicate text across different chunk ids
    unique: dict[str, Candidate] = {}
    for cand in sorted(merged.values(), key=lambda c: c.fusion_score, reverse=True):
        unique.setdefault(_fingerprint(cand.text), cand)

    out = list(unique.values())
    for cand in out:
        cand.fusion_score = round(cand.fusion_score, 6)
    return out[:limit]
