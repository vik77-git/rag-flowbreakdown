"""Document metadata registry (JSON on disk). Swappable for Postgres later."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class DocumentRegistry:
    def __init__(self, processed_dir: Path):
        self.path = processed_dir / "documents.json"
        self._lock = threading.Lock()
        self._docs: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._docs = json.loads(self.path.read_text())
            except ValueError:
                self._docs = {}

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._docs, indent=2))

    def upsert(self, document: dict) -> dict:
        with self._lock:
            self._docs[document["document_id"]] = document
            self._flush()
        return document

    def all(self) -> list[dict]:
        return sorted(self._docs.values(), key=lambda d: d.get("uploaded_at", 0), reverse=True)

    def get(self, document_id: str) -> dict | None:
        return self._docs.get(document_id)

    def delete(self, document_id: str) -> dict | None:
        with self._lock:
            doc = self._docs.pop(document_id, None)
            if doc:
                self._flush()
        for path in (self._text_path(document_id), self._chunk_path(document_id)):
            try:
                path.unlink()
            except OSError:
                pass
        return doc

    # -- chunk persistence (source of truth for rebuilding the indexes) -----
    def _text_path(self, document_id: str) -> Path:
        return self.path.parent / f"{document_id}.json"

    def _chunk_path(self, document_id: str) -> Path:
        return self.path.parent / f"{document_id}.chunks.json"

    def save_chunks(self, document_id: str, chunks: list[dict]) -> None:
        self._chunk_path(document_id).write_text(json.dumps(chunks, indent=2))

    def load_chunks(self, document_id: str) -> list[dict]:
        p = self._chunk_path(document_id)
        if not p.exists():
            return []
        try:
            rows = json.loads(p.read_text())
        except ValueError:
            return []
        return rows if isinstance(rows, list) else []

    def save_text(self, document_id: str, pages: list[dict]) -> None:
        (self.path.parent / f"{document_id}.json").write_text(json.dumps(pages, indent=2))

    def load_text(self, document_id: str) -> list[dict]:
        p = self.path.parent / f"{document_id}.json"
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text())
        except ValueError:
            return []
