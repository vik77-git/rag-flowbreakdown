from __future__ import annotations

from fastapi import APIRouter

from ..services import services

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict:
    s = services.settings
    return {
        "status": "ok",
        "embedding_model": s.embedding_model,
        "neural_embeddings": services.embedder.is_neural,
        "reranker_model": s.reranker_model,
        "models": s.allowed_models,
        "groq_configured": services.groq.rotator.configured,
        "groq_keys": services.groq.rotator.status(),
        "documents": len(services.registry.all()),
        "default_system_prompt": s.system_prompt,
        "max_upload_mb": s.max_upload_mb,
    }
