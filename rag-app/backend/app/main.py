from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import documents, health, query, session
from .services import services
from .utils.errors import RagError
from .utils.logging import log_event, setup_logging

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await services.startup()
    yield
    await services.shutdown()


app = FastAPI(title="RAG using LangChain", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RagError)
async def rag_error_handler(_: Request, exc: RagError) -> JSONResponse:
    log_event("request_error", message=exc.message, detail=exc.detail, status=exc.status_code)
    return JSONResponse(status_code=exc.status_code, content={"message": exc.message})


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    log_event("unhandled_error", error=type(exc).__name__)
    return JSONResponse(status_code=500, content={"message": "Something went wrong on the server."})


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(session.router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
