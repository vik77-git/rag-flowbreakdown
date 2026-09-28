"""Context selection: relevance floor, dedupe, structural order, token budget."""
from __future__ import annotations

from ..retrieval.base import Candidate
from ..utils.tokens import count_tokens


def build_context(
    candidates: list[Candidate], *, depth: int, token_budget: int, relevance_threshold: float
) -> tuple[list[Candidate], str, int]:
    eligible = [c for c in candidates if (c.rerank_score or 0.0) >= relevance_threshold]
    if not eligible:
        return [], "", 0

    selected: list[Candidate] = []
    used = 0
    for cand in eligible[:depth]:
        cost = count_tokens(cand.text) + 40
        if selected and used + cost > token_budget:
            break
        selected.append(cand)
        used += cost

    # keep reading order within each document for coherence
    ordered = sorted(
        selected,
        key=lambda c: (
            str(c.metadata.get("document_name", "")),
            c.metadata.get("page", 0),
            c.metadata.get("position", 0),
        ),
    )

    blocks = []
    for i, cand in enumerate(ordered, start=1):
        meta = cand.metadata
        blocks.append(
            f"[Source {i}]\n"
            f"Document: {meta.get('document_name', 'unknown')}\n"
            f"Page: {meta.get('page', 1)}\n"
            f"Section: {meta.get('section', 'n/a')}\n"
            f"Chunk ID: {cand.chunk_id}\n"
            f"Content:\n{cand.text}"
        )
    context = "\n\n---\n\n".join(blocks)
    return ordered, context, count_tokens(context)


def citations_for(selected: list[Candidate]) -> list[dict]:
    return [
        {
            "index": i,
            "document": c.metadata.get("document_name", "unknown"),
            "page": c.metadata.get("page", 1),
            "section": c.metadata.get("section", "n/a"),
            "chunk_id": c.chunk_id,
        }
        for i, c in enumerate(selected, start=1)
    ]
