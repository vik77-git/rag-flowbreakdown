from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.chunking.semantic import SemanticChunker  # noqa: E402
from app.embeddings.local import LocalEmbeddingProvider  # noqa: E402


class FakeEmbedder(LocalEmbeddingProvider):
    """Deterministic hashing embedder — no model download, works offline."""

    def __init__(self):
        self.model_name = "fake"
        self.batch_size = 8
        self._model = None
        self.dimension = 64


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def chunker(embedder) -> SemanticChunker:
    return SemanticChunker(embedder, threshold_percentile=90, min_chars=80, max_chars=400, overlap_sentences=0)


@pytest.fixture
def sample_text() -> str:
    return (ROOT / "sample_data" / "security_policy.txt").read_text()
