"""Local sentence-transformers embeddings, loaded once at startup.

If sentence-transformers is unavailable (e.g. offline test runs) a deterministic
hashing embedder is used instead, so the pipeline stays exercisable.
"""
from __future__ import annotations

import hashlib
import math


class LocalEmbeddingProvider:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self.dimension = 256

    @property
    def is_neural(self) -> bool:
        return self._model is not None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is not None:
            vectors = self._model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            return [list(map(float, v)) for v in vectors]
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    # -- fallback ----------------------------------------------------------
    def _hash_embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = [t for t in text.lower().split() if t]
        for token in tokens:
            h = int.from_bytes(hashlib.sha1(token.encode()).digest()[:8], "big")
            vec[h % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
