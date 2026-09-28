"""BM25 lexical retrieval.

Uses a smoothed (always non-negative) IDF: ``log(1 + (N - df + 0.5)/(df + 0.5))``.
Textbook Okapi IDF goes negative for terms that appear in more than half of the
corpus, which on a single small document (a handful of chunks) drives every
score to zero and makes retrieval return nothing. The smoothed form keeps common
terms weak instead of disqualifying them, so short corpora still rank correctly.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..chunking.semantic import Chunk
from .base import Candidate

_TOKEN = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Retriever:
    def __init__(self) -> None:
        # chunk_id -> (text, metadata, term frequencies, length)
        self._chunks: dict[str, tuple[str, dict, Counter, int]] = {}

    # -- writes ------------------------------------------------------------
    def add_documents(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            section = str(chunk.metadata.get("section") or "")
            tokens = tokenize(f"{chunk.text} {section}")
            self._chunks[chunk.chunk_id] = (chunk.text, chunk.metadata, Counter(tokens), len(tokens) or 1)

    def delete_document(self, document_id: str) -> None:
        for cid in [k for k, v in self._chunks.items() if v[1].get("document_id") == document_id]:
            self._chunks.pop(cid, None)

    def __len__(self) -> int:
        return len(self._chunks)

    # -- reads -------------------------------------------------------------
    def search(self, query: str, k: int, document_ids: list[str] | None = None) -> list[Candidate]:
        ids = [
            cid
            for cid, (_, meta, _, _) in self._chunks.items()
            if not document_ids or meta.get("document_id") in document_ids
        ]
        query_tokens = tokenize(query)
        if not ids or not query_tokens:
            return []

        n = len(ids)
        avgdl = sum(self._chunks[i][3] for i in ids) / n
        df = Counter()
        for cid in ids:
            df.update(self._chunks[cid][2].keys())

        scored: list[tuple[str, float]] = []
        for cid in ids:
            _, _, freqs, length = self._chunks[cid]
            score = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / (avgdl or 1)))
            if score > 0:
                scored.append((cid, score))

        scored.sort(key=lambda r: r[1], reverse=True)
        ranked = scored[:k]
        if not ranked:
            return []
        top = ranked[0][1] or 1.0
        return [
            Candidate(
                chunk_id=cid,
                text=self._chunks[cid][0],
                metadata=dict(self._chunks[cid][1]),
                bm25_score=round(score / top, 4),
                bm25_rank=rank,
            )
            for rank, (cid, score) in enumerate(ranked, start=1)
        ]
