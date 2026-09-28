from __future__ import annotations

from fastapi import Header

from ..reasoning.orchestrator import Orchestrator
from ..services import services

orchestrator = Orchestrator(services)


def session_id(x_session_id: str | None = Header(default=None)) -> str:
    return (x_session_id or "default")[:64]
