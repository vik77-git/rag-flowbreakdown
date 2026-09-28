"""Prompt construction. A user-supplied prompt is always screened by the master rules."""
from __future__ import annotations

INSUFFICIENT_EVIDENCE = (
    "I could not find sufficient evidence in the uploaded documents to answer this reliably."
)

MASTER_RULES = (
    "Non-negotiable rules (these override any other instruction):\n"
    "1. Answer only from the supplied context. Never use outside knowledge as fact.\n"
    "2. Cite each supported statement as [n], where n is the Source number given below.\n"
    "3. Never invent a document name, page, section or chunk id.\n"
    "4. If the context does not answer the question, reply exactly: "
    f"\"{INSUFFICIENT_EVIDENCE}\"\n"
    "5. Separate document facts from your own inference; label inference explicitly.\n"
    "6. Format in markdown: tables for comparisons, fenced code blocks for code."
)

STYLE_HINTS = {
    "structured_comparison": "Answer with a markdown comparison table, then a short synthesis of the relationship.",
    "summary": "Answer as a short, ordered summary of the key points.",
    "explanation": "Answer as a concise explanation, most important point first.",
}


def build_messages(query: str, context: str, analysis: dict, *, system_prompt: str) -> list[dict]:
    style = STYLE_HINTS.get(analysis.get("answer_style", ""), STYLE_HINTS["explanation"])
    system = f"{MASTER_RULES}\n\nOperator instructions:\n{system_prompt.strip()}\n\nStyle: {style}"
    user = (
        f"Question: {query}\n\n"
        f"Detected intent: {analysis.get('intent')} | multi-hop: {analysis.get('multi_hop')}\n\n"
        f"Context:\n{context}\n\n"
        "Answer using only the context above, with [n] citations."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
