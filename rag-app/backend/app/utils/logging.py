"""Structured JSON logging. Never logs document content or secrets."""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("rag")
    root.setLevel(logging.INFO)
    root.handlers = [handler]
    _configured = True


def log_event(event: str, **fields: Any) -> None:
    setup_logging()
    payload = {"ts": time.time(), "event": event}
    payload.update({k: v for k, v in fields.items() if v is not None})
    logging.getLogger("rag").info(json.dumps(payload, default=str))
