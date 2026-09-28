"""Central configuration. Everything is read from the environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_SYSTEM_PROMPT = (
    "You are an enterprise document analyst. Answer strictly from the supplied "
    "context. Cite every claim with [n] matching the given sources. If the context "
    "is insufficient, say so plainly. Never invent documents, pages or sections."
)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _path(name: str, default: str) -> Path:
    raw = os.environ.get(name) or default
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass
class Settings:
    groq_keys: list[str] = field(default_factory=list)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    model_20b: str = "openai/gpt-oss-20b"
    model_120b: str = "openai/gpt-oss-120b"
    requests_per_key: int = 5
    key_cooldown_seconds: float = 30.0
    groq_timeout: float = 120.0

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "openai/gpt-oss-20b"

    chroma_path: Path = ROOT / "data" / "chroma"
    upload_dir: Path = ROOT / "data" / "uploads"
    processed_dir: Path = ROOT / "data" / "processed"

    top_k_vector: int = 30
    top_k_bm25: int = 30
    top_k_rerank: int = 8
    threshold_percentile: float = 95.0
    min_chunk_chars: int = 200
    max_chunk_chars: int = 1800
    overlap_sentences: int = 1
    context_token_budget: int = 6000
    relevance_threshold: float = 0.25

    max_upload_mb: int = 25
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @property
    def allowed_models(self) -> list[str]:
        return [self.model_20b, self.model_120b]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    keys = [k.strip() for k in (os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_API_KEY_2")) if k and k.strip()]
    s = Settings(
        groq_keys=keys,
        groq_base_url=os.environ.get("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
        model_20b=os.environ.get("GROQ_MODEL_20B") or "openai/gpt-oss-20b",
        model_120b=os.environ.get("GROQ_MODEL_120B") or "openai/gpt-oss-120b",
        requests_per_key=_int("GROQ_REQUESTS_PER_KEY", 5),
        key_cooldown_seconds=_float("GROQ_KEY_COOLDOWN_SECONDS", 30.0),
        groq_timeout=_float("GROQ_TIMEOUT_SECONDS", 120.0),
        embedding_model=os.environ.get("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model=os.environ.get("RERANKER_MODEL") or "openai/gpt-oss-20b",
        chroma_path=_path("CHROMA_PATH", "./data/chroma"),
        upload_dir=_path("UPLOAD_DIR", "./data/uploads"),
        processed_dir=_path("PROCESSED_DIR", "./data/processed"),
        top_k_vector=_int("TOP_K_VECTOR", 30),
        top_k_bm25=_int("TOP_K_BM25", 30),
        top_k_rerank=_int("TOP_K_RERANK", 8),
        threshold_percentile=_float("SEMANTIC_THRESHOLD_PERCENTILE", 95.0),
        min_chunk_chars=_int("MIN_CHUNK_CHARS", 200),
        max_chunk_chars=_int("MAX_CHUNK_CHARS", 1800),
        overlap_sentences=_int("CHUNK_OVERLAP_SENTENCES", 1),
        context_token_budget=_int("CONTEXT_TOKEN_BUDGET", 6000),
        relevance_threshold=_float("RELEVANCE_THRESHOLD", 0.25),
        max_upload_mb=_int("MAX_UPLOAD_MB", 25),
        system_prompt=os.environ.get("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
    )
    for d in (s.chroma_path, s.upload_dir, s.processed_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s


settings = load_settings()
