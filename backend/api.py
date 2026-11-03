"""Thin HTTP layer over the document registry and the RAG pipeline.

Run from backend/ with: uvicorn api:app --port 8000

Starting this API reads document metadata and loads saved indexes. It never parses a
PDF or builds an index: those happen only after an explicit upload or reindex request.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Iterator, Literal

import groq
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import document_store
import rag_core

logger = logging.getLogger("api")

EngineState = Literal["loading", "ready", "failed"]

# The multipart wrapper adds a little to the file's own size.
UPLOAD_OVERHEAD_BYTES = 1024 * 1024


class SearchEngine:
    """Loads the embedding model and saved indexes on a background thread."""

    def __init__(self, documents: document_store.DocumentStore):
        self._documents = documents
        self._lock = threading.Lock()
        self.state: EngineState = "loading"
        self.stage = "starting"
        self.error: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None

    def start(self) -> None:
        threading.Thread(target=self._run, name="search-engine", daemon=True).start()

    def _on_stage(self, stage: str, details: dict[str, Any]) -> None:
        with self._lock:
            self.stage = stage

    def _run(self) -> None:
        try:
            self._documents.warm(self._on_stage)
        except Exception:  # Reported to the UI; the server stays up.
            logger.exception("Loading the search engine failed")
            with self._lock:
                self.state = "failed"
                self.error = "The search index could not be opened. Check the API logs."
                self.finished_at = time.time()
            return
        with self._lock:
            self.state = "ready"
            self.finished_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            end = self.finished_at or time.time()
            return {
                "state": self.state,
                "stage": self.stage if self.state == "loading" else None,
                "error": self.error,
                "elapsed_seconds": round(end - self.started_at),
            }


store = document_store.DocumentStore()
engine = SearchEngine(store)
pipeline = rag_core.RagPipeline()


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.load()
    engine.start()
    yield


app = FastAPI(title="RAG Document Assistant API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    selected_document_ids: list[str] = Field(default_factory=list, max_length=100)


def _error(status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "message": message, **extra})


def _not_found() -> JSONResponse:
    return _error(404, "document_not_found", "That document no longer exists.")


def _document_payload(record: document_store.DocumentRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "title": record.title,
        "status": record.status,
        "size_bytes": record.size_bytes,
        "pages_total": record.pages_total,
        "pages_indexed": record.pages_indexed,
        "pages_skipped": record.pages_skipped,
        "chunks": record.chunks,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "indexed_at": record.indexed_at,
        "suggested_questions": store.questions(record.id),
    }


@app.get("/api/workspace")
def workspace() -> dict[str, Any]:
    return {
        "index": engine.snapshot(),
        "documents": [_document_payload(record) for record in store.list()],
        "pipeline": {
            "chunk_size_chars": rag_core.CHUNK_SIZE,
            "chunk_overlap_chars": rag_core.CHUNK_OVERLAP,
            "embedding_model": rag_core.EMBEDDING_MODEL,
            "vector_store": "FAISS, one per document",
            "search_type": "MMR",
            "k": rag_core.SEARCH_K,
            "fetch_k": rag_core.SEARCH_FETCH_K,
            "lambda": rag_core.SEARCH_LAMBDA,
            "llm_model": rag_core.llm_model_name(),
        },
        "capabilities": {
            "upload": True,
            "document_selection": True,
            "conversation_memory": False,
        },
        "limits": {"max_upload_bytes": document_store.max_upload_bytes()},
    }


@app.get("/api/documents")
def list_documents() -> dict[str, Any]:
    return {"documents": [_document_payload(record) for record in store.list()]}


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    try:
        return _document_payload(store.get(document_id))
    except document_store.DocumentNotFound:
        return _not_found()


@app.post("/api/documents", status_code=201)
async def upload_document(request: Request):
    """Store a PDF, then queue it for indexing. The API stays available meanwhile."""
    limit = document_store.max_upload_bytes()
    declared = request.headers.get("content-length")
    # Checked first, so an oversized body isn't written to disk just to be rejected.
    if declared and declared.isdigit() and int(declared) > limit + UPLOAD_OVERHEAD_BYTES:
        return _error(413, "too_large", f"PDFs can be up to {limit // (1024 * 1024)} MB.")
    try:
        form = await request.form(max_files=1, max_fields=1)
    except Exception:
        return _error(400, "invalid_upload", "The upload couldn't be read.")
    file = form.get("file")
    if file is None or isinstance(file, str):
        return _error(400, "invalid_upload", "Choose a PDF to upload.")
    try:
        record = await run_in_threadpool(store.add_upload, file.file, file.filename, file.content_type)
    except document_store.UploadRejected as exc:
        return _error(exc.status_code, exc.code, exc.message, document_id=exc.document_id)
    finally:
        await form.close()
    store.request_indexing(record.id)
    return JSONResponse(status_code=201, content=_document_payload(record))


@app.post("/api/documents/{document_id}/reindex", status_code=202)
def reindex_document(document_id: str):
    try:
        record = store.request_indexing(document_id)
    except document_store.DocumentNotFound:
        return _not_found()
    except document_store.DocumentBusy:
        return _error(409, "document_busy", "This document is already being indexed.")
    return JSONResponse(status_code=202, content=_document_payload(record))


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    try:
        store.delete(document_id)
    except document_store.DocumentNotFound:
        return _not_found()
    except document_store.DocumentBusy:
        return _error(409, "document_busy", "This document can't be deleted while it's being indexed.")
    return Response(status_code=204)


def _event(event_type: str, **fields: Any) -> str:
    return json.dumps({"type": event_type, **fields}) + "\n"


def _llm_error(exc: Exception) -> dict[str, Any]:
    # Checked first, because a rejected key's error text also mentions "api_key".
    if isinstance(exc, groq.AuthenticationError):
        return {"code": "auth", "message": "The Groq API key was rejected."}
    # The client is created on the first question, so a missing key surfaces here.
    text = str(exc)
    if "GROQ_API_KEY" in text or "api_key" in text.lower():
        return {"code": "auth", "message": "GROQ_API_KEY is not set."}
    if isinstance(exc, groq.RateLimitError):
        retry_after = exc.response.headers.get("retry-after")
        return {
            "code": "rate_limited",
            "message": "The model provider's rate limit was reached.",
            "retry_after_seconds": round(float(retry_after)) if retry_after else None,
        }
    if isinstance(exc, groq.NotFoundError):
        return {
            "code": "model_unavailable",
            "message": f"The model {rag_core.llm_model_name()} is not available from Groq.",
        }
    if isinstance(exc, groq.APIConnectionError):
        return {"code": "network", "message": "The model provider could not be reached."}
    return {"code": "generation_failed", "message": "The answer could not be generated."}


def _run_query(document_ids: list[str], question: str) -> Iterator[str]:
    yield _event("stage", stage="retrieving")
    started = time.perf_counter()
    try:
        chunks = pipeline.retrieve(store.searchable(document_ids), question)
    except document_store.DocumentNotReady as exc:
        yield _event("error", stage="retrieving", code="document_not_ready", message=str(exc))
        return
    except Exception:
        yield _event(
            "error",
            stage="retrieving",
            code="retrieval_failed",
            message="The document search failed.",
        )
        return
    retrieval_ms = round((time.perf_counter() - started) * 1000)
    yield _event(
        "sources",
        sources=[asdict(chunk) for chunk in chunks],
        retrieval_ms=retrieval_ms,
    )

    yield _event("stage", stage="generating")
    started = time.perf_counter()
    answer = ""
    try:
        for text in pipeline.stream(question, chunks):
            answer += text
            yield _event("token", text=text)
    except Exception as exc:
        yield _event("error", stage="generating", **_llm_error(exc))
        return

    yield _event(
        "done",
        outcome="not_found" if rag_core.is_not_found(answer) else "answered",
        generation_ms=round((time.perf_counter() - started) * 1000),
        model=rag_core.llm_model_name(),
    )


@app.post("/api/query")
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        return _error(422, "empty_question", "Enter a question.")
    # Duplicates are dropped, and nothing outside the selection is ever searched.
    document_ids = list(dict.fromkeys(request.selected_document_ids))
    if not document_ids:
        return _error(400, "no_documents_selected", "Select at least one document to search.")
    for document_id in document_ids:
        try:
            record = store.get(document_id)
        except document_store.DocumentNotFound:
            return _not_found()
        if record.status != "ready":
            return _error(409, "document_not_ready", f"{record.name} isn't ready yet.", document_id=document_id)
    return StreamingResponse(_run_query(document_ids, question), media_type="application/x-ndjson")
