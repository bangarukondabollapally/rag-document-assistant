"""The document registry: stored PDFs, their metadata, and their saved indexes.

    backend/documents/<id>/
        original.pdf     The uploaded file. Its name never comes from the client.
        manifest.json     Filename, hash, status, page and chunk counts, settings.
        index/            That document's FAISS index.
        questions.json    Optional suggested questions.

No database is needed: the folder is the record. Loading the registry only reads
manifests and checks files, so it stays fast and never parses or embeds a document.
Indexing happens only when explicitly requested, on one background worker.

Command line use, from backend/: python document_store.py list | add <pdf> | reindex <id> | delete <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable

import rag_core

logger = logging.getLogger("document_store")

DEFAULT_ROOT = rag_core.ROOT / "documents"
DEFAULT_MAX_UPLOAD_MB = 50
PDF_NAME = "original.pdf"
MANIFEST_NAME = "manifest.json"
QUESTIONS_NAME = "questions.json"
INDEX_DIR_NAME = "index"
ACCEPTED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream", "", None}

_ID = re.compile(r"^[0-9a-f]{32}$")
_READ_SIZE = 1024 * 1024


class UploadRejected(Exception):
    """An upload can't be accepted. The message is safe to show to the user."""

    def __init__(self, status_code: int, code: str, message: str, document_id: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.document_id = document_id


class DocumentNotFound(Exception):
    pass


class DocumentBusy(Exception):
    """The document is queued or indexing, so it can't be changed yet."""


class DocumentNotReady(Exception):
    """The document can't be searched. The message says why, and is safe to show."""


def documents_root() -> Path:
    configured = os.environ.get("RAG_DOCUMENTS_DIR")
    return Path(configured) if configured else DEFAULT_ROOT


def max_upload_bytes() -> int:
    try:
        megabytes = float(os.environ.get("RAG_MAX_UPLOAD_MB") or DEFAULT_MAX_UPLOAD_MB)
    except ValueError:
        megabytes = DEFAULT_MAX_UPLOAD_MB
    return int(megabytes * 1024 * 1024)


def safe_display_name(filename: str | None) -> str:
    """Make a client filename safe to display. It's never used as a filesystem path."""
    name = re.split(r"[/\\]", filename or "")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip().strip(".")
    if not name:
        return "document.pdf"
    return name[:200] if len(name) <= 200 else f"{name[:196]}.pdf"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_questions(path: Path) -> list[dict[str, Any]]:
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
        return [{"question": str(item["question"]), "note": item.get("note")} for item in items]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        logger.warning("Ignoring invalid suggested questions in %s", path)
        return []


@dataclass
class DocumentRecord:
    id: str
    name: str
    sha256: str
    size_bytes: int = 0
    status: str = "uploaded"  # uploaded, indexing, ready or failed
    title: str | None = None
    pages_total: int | None = None
    pages_indexed: int | None = None
    pages_skipped: dict[str, int] = field(default_factory=dict)
    chunks: int | None = None
    settings: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    indexed_at: str | None = None


class DocumentStore:
    def __init__(self, root: Path | None = None, embeddings: Any = None):
        self.root = Path(root) if root else documents_root()
        self._lock = threading.RLock()
        self._records: dict[str, DocumentRecord] = {}
        self._questions: dict[str, list[dict[str, Any]]] = {}
        self._indexes: dict[str, Any] = {}
        self._index_lock = threading.Lock()
        self._embeddings = embeddings
        self._embeddings_lock = threading.Lock()
        self._busy: set[str] = set()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None

    # --- Paths ----------------------------------------------------------------

    def _directory(self, document_id: str) -> Path:
        # Only validated IDs reach the filesystem, which rules out path traversal.
        if not _ID.match(document_id):
            raise DocumentNotFound(document_id)
        return self.root / document_id

    def pdf_path(self, document_id: str) -> Path:
        return self._directory(document_id) / PDF_NAME

    def index_path(self, document_id: str) -> Path:
        return self._directory(document_id) / INDEX_DIR_NAME

    # --- Registry -------------------------------------------------------------

    def load(self) -> None:
        """Read every manifest. This parses no PDFs, embeds nothing and builds nothing."""
        records: dict[str, DocumentRecord] = {}
        questions: dict[str, list[dict[str, Any]]] = {}
        if self.root.is_dir():
            for directory in sorted(self.root.iterdir()):
                if directory.is_dir() and _ID.match(directory.name):
                    records[directory.name] = self._read_record(directory)
                    questions[directory.name] = _read_questions(directory / QUESTIONS_NAME)
        with self._lock:
            self._records = records
            self._questions = questions
            self._indexes = {}

    def _read_record(self, directory: Path) -> DocumentRecord:
        try:
            data = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
            known = {key: value for key, value in data.items() if key in DocumentRecord.__dataclass_fields__}
            record = DocumentRecord(**known)
            record.id = directory.name
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Unreadable manifest in %s: %s", directory.name, exc)
            return DocumentRecord(
                id=directory.name,
                name="Unreadable document",
                sha256="",
                status="failed",
                error="Its metadata couldn't be read. Reindex or delete it.",
            )
        # These checks change the reported status only: nothing is rebuilt automatically.
        problem = self._find_problem(directory, record)
        if problem:
            record.status, record.error = "failed", problem
        return record

    @staticmethod
    def _find_problem(directory: Path, record: DocumentRecord) -> str | None:
        if record.status in ("uploaded", "indexing"):
            return "Indexing was interrupted. Reindex to try again."
        if record.status != "ready":
            return None
        pdf = directory / PDF_NAME
        if not pdf.is_file():
            return "The stored PDF is missing. Delete it and upload it again."
        if rag_core.file_sha256(pdf) != record.sha256:
            return "The stored PDF has changed. Reindex to use it."
        if record.settings != rag_core.index_settings():
            return "The indexing settings have changed. Reindex to use it."
        index = directory / INDEX_DIR_NAME
        if not (index / "index.faiss").is_file() or not (index / "index.pkl").is_file():
            return "The saved index is missing. Reindex to rebuild it."
        return None

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda record: (record.created_at, record.id))

    def get(self, document_id: str) -> DocumentRecord:
        with self._lock:
            try:
                return self._records[document_id]
            except KeyError:
                raise DocumentNotFound(document_id) from None

    def questions(self, document_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._questions.get(document_id, [])

    def _save(self, record: DocumentRecord) -> None:
        record.updated_at = _now()
        directory = self._directory(record.id)
        temporary = directory / f"{MANIFEST_NAME}.tmp"
        temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        os.replace(temporary, directory / MANIFEST_NAME)

    def _find_by_hash(self, sha256: str) -> DocumentRecord | None:
        return next((record for record in self._records.values() if record.sha256 == sha256), None)

    # --- Adding documents -----------------------------------------------------

    def add_upload(self, stream: BinaryIO, filename: str | None, content_type: str | None = None) -> DocumentRecord:
        """Validate and store an upload as a new document. It isn't indexed yet."""
        name = safe_display_name(filename)
        if not name.lower().endswith(".pdf"):
            raise UploadRejected(415, "invalid_type", "Only PDF files can be uploaded.")
        if content_type not in ACCEPTED_CONTENT_TYPES:
            raise UploadRejected(415, "invalid_type", "Only PDF files can be uploaded.")

        limit = max_upload_bytes()
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=".incoming-"))
        try:
            digest = hashlib.sha256()
            size = 0
            with (staging / PDF_NAME).open("wb") as output:
                while block := stream.read(_READ_SIZE):
                    if size == 0 and not block.startswith(b"%PDF-"):
                        raise UploadRejected(415, "invalid_type", "That file isn't a valid PDF.")
                    size += len(block)
                    if size > limit:
                        raise UploadRejected(
                            413, "too_large", f"PDFs can be up to {limit // (1024 * 1024)} MB."
                        )
                    digest.update(block)
                    output.write(block)
            if size == 0:
                raise UploadRejected(415, "invalid_type", "That file is empty.")

            sha256 = digest.hexdigest()
            # Held while registering, so two identical uploads can't both be accepted.
            with self._lock:
                existing = self._find_by_hash(sha256)
                if existing:
                    raise UploadRejected(
                        409, "duplicate", f"This PDF is already uploaded as {existing.name}.", existing.id
                    )
                record = DocumentRecord(id=uuid.uuid4().hex, name=name, sha256=sha256, size_bytes=size)
                (staging / MANIFEST_NAME).write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
                staging.rename(self._directory(record.id))
                self._records[record.id] = record
                self._questions[record.id] = []
            return record
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def add_file(self, path: Path) -> DocumentRecord:
        """Register a local PDF, along with any <name>.questions.json beside it."""
        with path.open("rb") as stream:
            record = self.add_upload(stream, path.name, "application/pdf")
        sidecar = path.with_suffix(".questions.json")
        if sidecar.is_file():
            shutil.copyfile(sidecar, self._directory(record.id) / QUESTIONS_NAME)
            with self._lock:
                self._questions[record.id] = _read_questions(self._directory(record.id) / QUESTIONS_NAME)
        return record

    # --- Indexing -------------------------------------------------------------

    def request_indexing(self, document_id: str) -> DocumentRecord:
        """Queue one document for an explicit build. Other documents aren't affected."""
        with self._lock:
            record = self.get(document_id)
            if document_id in self._busy:
                raise DocumentBusy(document_id)
            record.status = "uploaded"
            record.error = None
            self._indexes.pop(document_id, None)
            self._save(record)
            self._busy.add(document_id)
            self._queue.put(document_id)
            self._ensure_worker()
            return record

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            # A daemon, so stopping the server doesn't wait for a long build. The document
            # then reports "interrupted" on the next start rather than rebuilding itself.
            self._worker = threading.Thread(target=self._work, name="document-indexer", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            document_id = self._queue.get()
            try:
                self.index_now(document_id)
            except Exception:
                logger.exception("Unexpected failure indexing %s", document_id)
            finally:
                with self._lock:
                    self._busy.discard(document_id)
                self._queue.task_done()

    def wait_until_idle(self) -> None:
        self._queue.join()

    def index_now(self, document_id: str, on_stage: rag_core.StageCallback | None = None) -> DocumentRecord:
        """Parse, chunk and embed one document, then save its index and record the result."""
        with self._lock:
            record = self.get(document_id)
            record.status = "indexing"
            record.error = None
            self._save(record)
        pdf = self.pdf_path(document_id)
        try:
            sha256 = rag_core.file_sha256(pdf) if pdf.is_file() else record.sha256
            vectorstore, stats = rag_core.build_index(
                pdf,
                self.index_path(document_id),
                self.embeddings(),
                name=record.name,
                on_stage=on_stage or rag_core._no_op,
            )
        except rag_core.DocumentError as exc:
            return self._record_failure(record, str(exc))
        except Exception:
            logger.exception("Indexing %s failed", record.name)
            return self._record_failure(record, "Indexing failed. Check the API logs, then try again.")

        with self._lock:
            record.status = "ready"
            record.sha256 = sha256
            record.title = stats.title
            record.pages_total = stats.pages_total
            record.pages_indexed = stats.pages_indexed
            record.pages_skipped = stats.pages_skipped
            record.chunks = stats.chunks
            record.settings = rag_core.index_settings()
            record.indexed_at = _now()
            record.error = None
            self._save(record)
            self._indexes[document_id] = vectorstore
        return record

    def _record_failure(self, record: DocumentRecord, message: str) -> DocumentRecord:
        with self._lock:
            record.status = "failed"
            record.error = message
            self._indexes.pop(record.id, None)
            self._save(record)
        return record

    # --- Opening saved indexes ------------------------------------------------

    def embeddings(self) -> Any:
        with self._embeddings_lock:
            if self._embeddings is None:
                self._embeddings = rag_core._make_embeddings()
            return self._embeddings

    def open_index(self, document_id: str) -> Any:
        """Return a document's saved index, loading it once. It never builds one."""
        record = self.get(document_id)
        if record.status != "ready":
            raise DocumentNotReady(f"{record.name} isn't ready.")
        embeddings = self.embeddings()
        with self._index_lock:
            if document_id in self._indexes:
                return self._indexes[document_id]
            try:
                vectorstore = rag_core.load_index(self.index_path(document_id), embeddings, record.chunks or 0)
            except rag_core.IndexLoadError as exc:
                with self._lock:
                    record.status = "failed"
                    record.error = f"{exc} Reindex to rebuild it."
                raise DocumentNotReady(f"{record.name}: {exc}") from exc
            self._indexes[document_id] = vectorstore
            return vectorstore

    def warm(self, on_stage: rag_core.StageCallback | None = None) -> None:
        """Load the model and saved indexes for ready documents. A damaged one is only reported."""
        report = on_stage or rag_core._no_op
        ready = [record for record in self.list() if record.status == "ready"]
        if not ready:
            return
        report("loading_model", {})
        self.embeddings()
        report("opening_indexes", {})
        for record in ready:
            try:
                self.open_index(record.id)
            except DocumentNotReady as exc:
                logger.warning("%s", exc)

    def searchable(self, document_ids: list[str]) -> list[rag_core.SearchableDocument]:
        """Return exactly the requested documents, in a form retrieval can search."""
        return [
            rag_core.SearchableDocument(document_id, self.get(document_id).name, self.open_index(document_id))
            for document_id in document_ids
        ]

    # --- Deleting -------------------------------------------------------------

    def delete(self, document_id: str) -> None:
        """Remove a document's PDF, metadata and index together."""
        with self._lock:
            self.get(document_id)
            if document_id in self._busy:
                raise DocumentBusy(document_id)
            directory = self._directory(document_id)
            shutil.rmtree(directory)
            self._records.pop(document_id, None)
            self._questions.pop(document_id, None)
            self._indexes.pop(document_id, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage indexed documents from the command line.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List documents and their status.")
    commands.add_parser("add", help="Add a PDF and index it.").add_argument("path", type=Path)
    commands.add_parser("reindex", help="Rebuild one document's index.").add_argument("id")
    commands.add_parser("delete", help="Delete a document and its index.").add_argument("id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    store = DocumentStore()
    store.load()
    try:
        if args.command == "add":
            record = store.index_now(store.add_file(args.path).id, lambda stage, details: print(stage, details or ""))
            print(f"{record.id}  {record.status}  {record.error or ''}")
        elif args.command == "reindex":
            record = store.index_now(args.id, lambda stage, details: print(stage, details or ""))
            print(f"{record.id}  {record.status}  {record.error or ''}")
        elif args.command == "delete":
            store.delete(args.id)
            print("Deleted.")
        else:
            for record in store.list():
                print(f"{record.id}  {record.status:8}  {record.chunks or 0:>6} chunks  {record.name}")
    except (UploadRejected, DocumentNotFound) as exc:
        raise SystemExit(f"{exc}") from exc


if __name__ == "__main__":
    main()
