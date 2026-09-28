from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, UploadFile

from ..ingestion.pipeline import SUPPORTED_EXTENSIONS
from ..schemas.models import DocumentSummary
from ..services import services
from ..storage.files import validate_upload
from ..utils.errors import RagError

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
def list_documents() -> list[dict]:
    return services.registry.all()


@router.post("/upload", response_model=DocumentSummary)
async def upload_document(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    safe = validate_upload(
        file.filename or "document",
        content,
        file.content_type,
        services.settings.max_upload_bytes,
        SUPPORTED_EXTENSIONS,
    )
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: services.ingest_file(content, safe))


@router.get("/{document_id}")
def get_document(document_id: str) -> dict:
    doc = services.registry.get(document_id)
    if not doc:
        raise RagError("Document not found.", 404)
    chunks = services.vector_store.get_document_chunks(document_id)
    sections: list[dict] = []
    seen: set[str] = set()
    for c in chunks:
        title = c.metadata.get("section")
        if title and title not in seen:
            seen.add(title)
            sections.append({"title": title, "page": c.metadata.get("page", 1)})
    return {
        **doc,
        "sections": sections,
    }


@router.get("/{document_id}/pages/{page_number}")
def get_document_page(document_id: str, page_number: int) -> dict:
    if not services.registry.get(document_id):
        raise RagError("Document not found.", 404)
    pages = services.registry.load_text(document_id)
    page = next((p for p in pages if int(p.get("page", 0)) == page_number), None)
    if not page:
        raise RagError("Page not found.", 404)
    return page


@router.delete("/{document_id}")
def delete_document(document_id: str) -> dict:
    if not services.delete_document(document_id):
        raise RagError("Document not found.", 404)
    return {"deleted": document_id}
