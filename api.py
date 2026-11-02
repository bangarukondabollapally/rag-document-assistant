"""Thin HTTP adapter over rag_core for the React workspace.

Run with: uvicorn api:app --port 8000
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Iterator, Literal

import groq
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import rag_core

IndexState = Literal["loading", "ready", "not_indexed", "failed"]

REBUILD_COMMAND = "python rag_core.py --rebuild"


def _open_default_pipeline(on_stage: rag_core.StageCallback) -> rag_core.RagPipeline:
    return rag_core.open_saved_pipeline(on_stage=on_stage)


class IndexManager:
    """Opens the saved index on a background thread. It never builds an index."""

    def __init__(self, opener=_open_default_pipeline):
        self._opener = opener
        self._lock = threading.Lock()
        self.state: IndexState = "loading"
        self.stage = "checking"
        self.reason: str | None = None
        self.pages: int | None = None
        self.chunks: int | None = None
        self.error: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.pipeline: rag_core.RagPipeline | None = None

    def start(self) -> None:
        threading.Thread(target=self._run, name="rag-index", daemon=True).start()

    def _on_stage(self, stage: str, details: dict[str, Any]) -> None:
        with self._lock:
            self.stage = stage
            self.pages = details.get("pages", self.pages)
            self.chunks = details.get("chunks", self.chunks)

    def _run(self) -> None:
        try:
            pipeline = self._opener(self._on_stage)
        except rag_core.IndexNotAvailable as exc:
            with self._lock:
                self.state = "not_indexed"
                self.reason = str(exc)
                self.finished_at = time.time()
            return
        except Exception as exc:  # Reported to the UI; the server stays up.
            with self._lock:
                self.state = "failed"
                self.error = _index_error_message(exc)
                self.finished_at = time.time()
            return
        with self._lock:
            self.pipeline = pipeline
            self.state = "ready"
            self.finished_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            end = self.finished_at or time.time()
            return {
                "state": self.state,
                "stage": self.stage if self.state == "loading" else None,
                "reason": self.reason,
                "rebuild_command": REBUILD_COMMAND if self.state == "not_indexed" else None,
                "pages": self.pages,
                "chunks": self.chunks,
                "error": self.error,
                "elapsed_seconds": round(end - self.started_at),
            }


def _index_error_message(exc: Exception) -> str:
    if isinstance(exc, rag_core.DocumentError):
        return str(exc)
    return "The saved index could not be opened. Check the API logs."


manager = IndexManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.start()
    yield


app = FastAPI(title="RAG Document Assistant API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _document_payload(index: dict[str, Any], name: str) -> dict[str, Any]:
    pipeline = manager.pipeline
    info = pipeline.document if pipeline else None
    return {
        "id": name,
        "name": name,
        "title": info.title if info else None,
        "status": index["state"],
        "pages_total": info.pages_total if info else None,
        "pages_indexed": info.pages_indexed if info else index["pages"],
        "pages_skipped": info.pages_skipped if info else {},
        "chunks": info.chunks if info else index["chunks"],
    }


@app.get("/api/workspace")
def workspace() -> dict[str, Any]:
    index = manager.snapshot()
    document = rag_core.resolve_document()
    return {
        "index": index,
        "documents": [_document_payload(index, document.name)],
        "suggested_questions": rag_core.suggested_questions(document),
        "pipeline": {
            "chunk_size_chars": rag_core.CHUNK_SIZE,
            "chunk_overlap_chars": rag_core.CHUNK_OVERLAP,
            "embedding_model": rag_core.EMBEDDING_MODEL,
            "vector_store": "FAISS (saved on disk)",
            "search_type": "MMR",
            "k": rag_core.SEARCH_K,
            "fetch_k": rag_core.SEARCH_FETCH_K,
            "lambda": rag_core.SEARCH_LAMBDA,
            "llm_model": rag_core.llm_model_name(),
        },
        "capabilities": {
            "upload": False,
            "document_selection": False,
            "conversation_memory": False,
        },
    }


def _event(event_type: str, **fields: Any) -> str:
    return json.dumps({"type": event_type, **fields}) + "\n"


def _llm_error(exc: Exception) -> dict[str, Any]:
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
    if isinstance(exc, groq.AuthenticationError):
        return {"code": "auth", "message": "The Groq API key was rejected."}
    if isinstance(exc, groq.NotFoundError):
        return {
            "code": "model_unavailable",
            "message": f"The model {rag_core.llm_model_name()} is not available from Groq.",
        }
    if isinstance(exc, groq.APIConnectionError):
        return {"code": "network", "message": "The model provider could not be reached."}
    return {"code": "generation_failed", "message": "The answer could not be generated."}


def _run_query(pipeline: rag_core.RagPipeline, question: str) -> Iterator[str]:
    yield _event("stage", stage="retrieving")
    started = time.perf_counter()
    try:
        chunks = pipeline.retrieve(question)
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
        return JSONResponse(
            status_code=422,
            content={"code": "empty_question", "message": "Enter a question."},
        )
    pipeline = manager.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"code": "not_ready", "message": "The document index isn't ready."},
        )
    return StreamingResponse(_run_query(pipeline, question), media_type="application/x-ndjson")
