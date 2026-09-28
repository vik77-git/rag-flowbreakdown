"""Heuristic query analyzer. No LLM call for ordinary queries."""
from __future__ import annotations

import re

from ..retrieval.bm25 import tokenize

_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "what", "which", "does", "do", "for",
    "on", "about", "with", "how", "why", "when", "who", "any", "there", "this", "that", "say", "says",
    "document", "documents", "me", "please", "can", "could", "list", "give",
}
_COMPARE = ("compare", "versus", " vs ", "difference", "differences", "relationship", "contrast", "between")
_SUMMARY = ("summarize", "summary", "overview", "outline", "recap")
_MULTI = ("and also", "as well as", "both", "relate", "relationship", "across", "each")
_ENTITY = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b")


def analyze_query(query: str, *, base_depth: int = 5) -> dict:
    text = query.strip()
    lowered = f" {text.lower()} "
    keywords = [t for t in dict.fromkeys(tokenize(text)) if t not in _STOP and len(t) > 2]
    comparison = any(k in lowered for k in _COMPARE)
    summary = any(k in lowered for k in _SUMMARY)
    multi_hop = comparison or any(k in lowered for k in _MULTI) or lowered.count("?") > 1

    if multi_hop or summary:
        depth = max(8, base_depth + 3)
    elif len(keywords) <= 3 and len(text.split()) <= 8:
        depth = max(3, base_depth - 2)
    else:
        depth = base_depth

    if comparison:
        intent, style = "comparison", "structured_comparison"
    elif summary:
        intent, style = "summarization", "summary"
    elif lowered.strip().startswith(("how ", "why ")):
        intent, style = "explanation", "explanation"
    else:
        intent, style = "fact_extraction", "explanation"

    return {
        "intent": intent,
        "keywords": keywords[:10],
        "entities": [e for e in _ENTITY.findall(text) if e.lower() not in _STOP][:6],
        "comparison": comparison,
        "multi_hop": multi_hop,
        "retrieval_depth": depth,
        "answer_style": style,
    }
