"""User-safe error type. Detailed causes stay in the logs."""
from __future__ import annotations


class RagError(Exception):
    """An error whose message is safe to show in the UI."""

    def __init__(self, message: str, status_code: int = 400, *, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail  # logged only, never returned
