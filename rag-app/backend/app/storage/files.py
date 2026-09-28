"""Safe upload storage: sanitized names, traversal-proof paths, size limits."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from ..utils.errors import RagError

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

ALLOWED_MIME = {
    ".pdf": {"application/pdf", "application/octet-stream", ""},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
        "",
    },
    ".txt": {"text/plain", "application/octet-stream", ""},
    ".md": {"text/plain", "text/markdown", "application/octet-stream", ""},
}
MAGIC = {".pdf": b"%PDF", ".docx": b"PK"}


def sanitize_filename(name: str) -> str:
    base = Path(name or "document").name
    base = _SAFE.sub("_", base).strip("._") or "document"
    return base[:120]


def validate_upload(filename: str, content: bytes, content_type: str | None, max_bytes: int, allowed_ext) -> str:
    safe = sanitize_filename(filename)
    ext = Path(safe).suffix.lower()
    if ext not in allowed_ext:
        raise RagError(f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(allowed_ext)}.", 415)
    if not content:
        raise RagError("The uploaded file is empty.", 422)
    if len(content) > max_bytes:
        raise RagError(f"File is too large. Maximum size is {max_bytes // (1024 * 1024)} MB.", 413)
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime and mime not in ALLOWED_MIME.get(ext, set()):
        raise RagError("The file content does not match its extension.", 415)
    magic = MAGIC.get(ext)
    if magic and not content.startswith(magic):
        raise RagError("The file content does not match its extension.", 415)
    if ext in (".txt", ".md") and not content.decode("utf-8", errors="ignore").strip():
        raise RagError("The uploaded file contains no text.", 422)
    return safe


def store_upload(upload_dir: Path, safe_name: str, content: bytes) -> tuple[str, Path]:
    document_id = f"doc_{uuid.uuid4().hex[:10]}"
    target = (upload_dir / f"{document_id}__{safe_name}").resolve()
    if upload_dir.resolve() not in target.parents:
        raise RagError("Invalid upload path.", 400)
    target.write_bytes(content)
    return document_id, target


def resolve_stored(upload_dir: Path, stored_name: str) -> Path:
    target = (upload_dir / Path(stored_name).name).resolve()
    if upload_dir.resolve() not in target.parents or not target.exists():
        raise RagError("Document file not found.", 404)
    return target
