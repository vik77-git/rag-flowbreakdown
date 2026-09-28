"""Semantic chunking driven by consecutive sentence-embedding distance."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..utils.tokens import count_tokens

if TYPE_CHECKING:  # pragma: no cover
    from ..embeddings.base import EmbeddingProvider
    from ..ingestion.structure import Section

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (pct / 100.0) * (len(ordered) - 1)
    low, high = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (idx - low)


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 1.0
    return 1.0 - dot / (na * nb)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    def __init__(
        self,
        embedder: "EmbeddingProvider",
        *,
        threshold_percentile: float = 95.0,
        min_chars: int = 200,
        max_chars: int = 1800,
        overlap_sentences: int = 1,
    ):
        self.embedder = embedder
        self.threshold_percentile = threshold_percentile
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_sentences = max(0, overlap_sentences)

    # -- boundary detection -------------------------------------------------
    def _groups(self, sentences: list[str]) -> list[list[str]]:
        if len(sentences) <= 1:
            return [sentences] if sentences else []
        vectors = self.embedder.embed_documents(sentences)
        distances = [cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        threshold = percentile(distances, self.threshold_percentile)

        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]
        for i, sentence in enumerate(sentences[1:]):
            size = sum(len(s) + 1 for s in current)
            hard_split = size + len(sentence) > self.max_chars
            semantic_split = distances[i] >= threshold and size >= self.min_chars
            if hard_split or semantic_split:
                groups.append(current)
                current = current[-self.overlap_sentences:] if self.overlap_sentences and not hard_split else []
                current.append(sentence)
            else:
                current.append(sentence)
        if current:
            groups.append(current)
        return self._merge_small(groups)

    def _merge_small(self, groups: list[list[str]]) -> list[list[str]]:
        merged: list[list[str]] = []
        for group in groups:
            text_len = sum(len(s) + 1 for s in group)
            if merged and text_len < self.min_chars:
                candidate = sum(len(s) + 1 for s in merged[-1]) + text_len
                if candidate <= self.max_chars:
                    merged[-1].extend(group)
                    continue
            merged.append(list(group))
        return merged

    # -- public API ---------------------------------------------------------
    def chunk_section(
        self, section: "Section", document_id: str, document_name: str, *, start_position: int = 0
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        def emit(text: str, chunk_type: str) -> None:
            text = text.strip()
            if not text:
                return
            position = start_position + len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_c{position:04d}",
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "document_name": document_name,
                        "page": section.page,
                        "section": section.title,
                        "parent_section": section.parent,
                        "chunk_type": chunk_type,
                        "position": position,
                        "token_count": count_tokens(text),
                    },
                )
            )

        prose: list[str] = []
        for block in section.blocks:
            if block.block_type == "table":
                for group in self._groups(split_sentences("\n".join(prose))) if prose else []:
                    emit(" ".join(group), "paragraph")
                prose = []
                emit(f"{section.title}\n{block.text}", "table")
            else:
                prose.append(block.text)
        for group in self._groups(split_sentences("\n".join(prose))) if prose else []:
            emit(" ".join(group), "paragraph")
        return chunks
