from __future__ import annotations

from fastapi import APIRouter, Depends

from ..schemas.models import SessionUsage
from ..services import services
from .deps import session_id

router = APIRouter(prefix="/api/session", tags=["session"])


@router.get("/usage", response_model=SessionUsage)
def usage(sid: str = Depends(session_id)) -> dict:
    stats = services.sessions.get(sid)
    queries = stats["queries"]
    return {
        **stats,
        "avg_tokens_per_query": round(stats["total_tokens"] / queries, 1) if queries else 0.0,
        "documents": len(services.registry.all()),
    }
