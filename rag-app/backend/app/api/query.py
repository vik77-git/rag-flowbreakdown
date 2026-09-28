from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..schemas.models import QueryRequest, QueryResponse
from ..services import services
from ..utils.errors import RagError
from ..utils.logging import log_event
from .deps import orchestrator, session_id

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def run_query(payload: QueryRequest, sid: str = Depends(session_id)) -> dict:
    final: dict | None = None
    async for event in orchestrator.run(
        payload.query,
        document_ids=payload.document_ids,
        model=payload.model,
        system_prompt=payload.system_prompt,
        session_id=sid,
    ):
        if event["event"] == "query_complete":
            final = event["data"]
    if final is None:
        raise RagError("The query could not be completed.", 500)
    return final


@router.post("/stream")
async def stream_query(payload: QueryRequest, sid: str = Depends(session_id)) -> StreamingResponse:
    async def generator():
        try:
            async for event in orchestrator.run(
                payload.query,
                document_ids=payload.document_ids,
                model=payload.model,
                system_prompt=payload.system_prompt,
                session_id=sid,
            ):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        except RagError as exc:
            log_event("query_failed", message=exc.message, detail=exc.detail)
            yield f"event: error\ndata: {json.dumps({'message': exc.message})}\n\n"
        except Exception as exc:  # pragma: no cover
            log_event("query_failed", error=type(exc).__name__)
            yield "event: error\ndata: {\"message\": \"Something went wrong while answering.\"}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/{query_id}", response_model=QueryResponse)
def get_query(query_id: str) -> dict:
    stored = services.sessions.queries.get(query_id)
    if not stored:
        raise RagError("Query not found in this application session.", 404)
    return stored
