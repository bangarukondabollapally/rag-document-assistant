"""The RAG pipeline, independent of any UI and of any particular document.

    document → load → select content pages → chunk → embed → saved FAISS index
                                                                   ↓
                              answer ← LLM ← context ← retrieve ← load index

Building an index and opening one are separate operations. Applications call
open_saved_pipeline, which never builds. Only rebuild_index parses, embeds and saves.

Run `python rag_core.py` to see the index status, or add --rebuild to build the index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any, Callable, Iterator

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

logger = logging.getLogger("rag_core")

# A local development document, not committed, is used when present.
DEV_DOCUMENT = ROOT / "DocumentLoaders" / "deeplearning.pdf"
SAMPLE_DOCUMENT = ROOT / "samples" / "rag-guide.pdf"
DEFAULT_INDEX_ROOT = ROOT / ".rag_index"

CHUNK_SIZE = 500  # characters, not tokens
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SEARCH_K = 6
SEARCH_FETCH_K = 20
SEARCH_LAMBDA = 0.7
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"

# Increase when page selection changes, so saved indexes are rebuilt.
INGEST_VERSION = 1

NOT_FOUND_MESSAGE = "I couldn't find that in the document."

PROMPT_MESSAGES = [
    ("system", """You are a helpful AI assistant answering questions about a document.
    Answer questions clearly and thoroughly based strictly on the context provided.
    If the answer is in the context, explain it well with any relevant details.
    If the question is completely outside the context, say: "I couldn't find that in the document."
    Do not make up information."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
]

StageCallback = Callable[[str, dict[str, Any]], None]


class DocumentError(Exception):
    """A document can't be used. The message is safe to show to the user."""


class IndexLoadError(Exception):
    """A saved index exists but can't be used."""


class IndexNotAvailable(Exception):
    """There's no usable saved index. The message says why, and is safe to show."""


def _no_op(stage: str, details: dict[str, Any]) -> None:
    pass


@cache
def _prompt() -> Any:
    # Imported here, like the other heavy libraries, so that importing this module and
    # binding the server stay fast. langchain_core.prompts loads transformers.
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(PROMPT_MESSAGES)


def llm_model_name() -> str:
    return os.environ.get("RAG_LLM_MODEL") or DEFAULT_LLM_MODEL


def index_root() -> Path:
    configured = os.environ.get("RAG_INDEX_DIR")
    return Path(configured) if configured else DEFAULT_INDEX_ROOT


def resolve_document(document: str | Path | None = None) -> Path:
    """Choose the document: the argument, then RAG_DOCUMENT_PATH, then the defaults."""
    chosen = document or os.environ.get("RAG_DOCUMENT_PATH")
    if chosen:
        path = Path(chosen)
        return path if path.is_absolute() else ROOT / path
    return DEV_DOCUMENT if DEV_DOCUMENT.is_file() else SAMPLE_DOCUMENT


def suggested_questions(document: Path) -> list[dict[str, str | None]]:
    """Read optional questions from <name>.questions.json beside the document."""
    sidecar = document.with_suffix(".questions.json")
    try:
        items = json.loads(sidecar.read_text(encoding="utf-8"))
        return [{"question": str(item["question"]), "note": item.get("note")} for item in items]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        logger.warning("Ignoring invalid suggested questions in %s", sidecar.name)
        return []


# --- Document loading ---------------------------------------------------------


def load_document(path: Path) -> list[Document]:
    """Read a PDF into one Document per page, keeping page numbers in the metadata."""
    if not path.is_file():
        raise DocumentError(
            f"The document {path.name} was not found. Set RAG_DOCUMENT_PATH in .env to a PDF."
        )
    from langchain_community.document_loaders import PyPDFLoader

    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:
        raise DocumentError(f"{path.name} could not be read as a PDF.") from exc
    for page in pages:
        page.metadata["source"] = path.name
    return pages


# --- Page selection -----------------------------------------------------------

_TOC_LINE = re.compile(r"[^\W\d].*(?:\s|\.\s?)\d{1,4}$")
_INDEX_LINE = re.compile(r",\s*\d{1,4}(?:\s*[-–,]\s*\d{1,4})*$")
_ISBN = re.compile(r"\bisbn\b|\b97[89][- ]\d", re.IGNORECASE)

# Each share is high enough, and each minimum long enough, that ordinary prose, code,
# slides and numeric tables aren't removed. A missed page costs little; a wrongly
# removed page makes its content unanswerable.
_TOC_MIN_LINES = 8
_TOC_MIN_SHARE = 0.8
_INDEX_MIN_LINES = 20
_INDEX_MIN_SHARE = 0.5


def classify_page(text: str) -> str | None:
    """Return why a page isn't document content, or None for a content page."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "blank"

    def share(pattern: re.Pattern[str]) -> float:
        return sum(bool(pattern.search(line)) for line in lines) / len(lines)

    # Checked first, because an index line also has the shape of a contents line.
    if len(lines) >= _INDEX_MIN_LINES and share(_INDEX_LINE) >= _INDEX_MIN_SHARE:
        return "back_of_book_index"
    if len(lines) >= _TOC_MIN_LINES and share(_TOC_LINE) >= _TOC_MIN_SHARE:
        return "table_of_contents"
    if "all rights reserved" in text.lower() and _ISBN.search(text):
        return "copyright"
    return None


def select_content_pages(pages: list[Document]) -> tuple[list[Document], dict[str, int]]:
    """Keep content pages by what they contain, never by where they are in the file."""
    kept: list[Document] = []
    skipped: dict[str, int] = {}
    for page in pages:
        reason = classify_page(page.page_content)
        if reason is None:
            kept.append(page)
        else:
            skipped[reason] = skipped.get(reason, 0) + 1
    return kept, skipped


# --- Chunking -----------------------------------------------------------------


def chunk_pages(pages: list[Document]) -> list[Document]:
    # Imported here because it takes seconds to load, and only a build needs it.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return [chunk for chunk in splitter.split_documents(pages) if chunk.page_content.strip()]


# --- Saved index --------------------------------------------------------------


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    title: str | None
    pages_total: int
    pages_indexed: int
    chunks: int
    pages_skipped: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexCheck:
    status: str  # "valid", "missing" or "stale"
    directory: Path
    reason: str | None = None
    manifest: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _settings() -> dict[str, Any]:
    """Everything that changes an index's contents. A difference requires a rebuild."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "ingest_version": INGEST_VERSION,
    }


def _index_directory(document_hash: str) -> Path:
    return index_root() / document_hash[:32]


def check_index(document: Path) -> IndexCheck:
    """Report whether a usable saved index exists, without loading it."""
    if not document.is_file():
        raise DocumentError(
            f"The document {document.name} was not found. Set RAG_DOCUMENT_PATH in .env to a PDF."
        )
    document_hash = _sha256(document)
    directory = _index_directory(document_hash)
    # The manifest is written last, so an interrupted build never has one.
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return IndexCheck("missing", directory, "No saved index exists for this document.")
    except (OSError, ValueError):
        return IndexCheck("stale", directory, "The saved index's manifest couldn't be read.")

    if manifest.get("document_sha256") != document_hash:
        return IndexCheck("stale", directory, "The document has changed.")
    if manifest.get("settings") != _settings():
        return IndexCheck("stale", directory, "The indexing settings have changed.")
    return IndexCheck("valid", directory, manifest=manifest)


def _make_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _make_llm() -> Any:
    # Checked here because a blank key otherwise surfaces as a misleading connection error.
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise ValueError("GROQ_API_KEY is not set.")

    from langchain_groq import ChatGroq

    return ChatGroq(model=llm_model_name())


def _info_from_manifest(document: Path, manifest: dict[str, Any]) -> DocumentInfo:
    # The name comes from the current path, so renaming a file doesn't need a rebuild.
    return DocumentInfo(**{**manifest["document"], "name": document.name})


def build_index(
    document: Path,
    embeddings: Embeddings,
    on_stage: StageCallback = _no_op,
) -> tuple[FAISS, DocumentInfo]:
    """Create and save an index. Replaces any earlier index only after a full build."""
    on_stage("loading", {})
    pages = load_document(document)
    kept, skipped = select_content_pages(pages)
    chunks = chunk_pages(kept)
    if not chunks:
        raise DocumentError(
            f"{document.name} has no extractable text. Scanned PDFs aren't supported yet."
        )

    title = pages[0].metadata.get("title") or None
    info = DocumentInfo(document.name, title, len(pages), len(kept), len(chunks), skipped)
    logger.info("Indexing %s: kept %d of %d pages, skipped %s", document.name, len(kept), len(pages), skipped)
    on_stage("embedding", {"chunks": len(chunks), "pages": len(kept)})

    vectorstore = FAISS.from_documents(chunks, embeddings)

    document_hash = _sha256(document)
    directory = _index_directory(document_hash)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".building-"))
    try:
        vectorstore.save_local(str(temporary))
        manifest = {
            "document_sha256": document_hash,
            "settings": _settings(),
            "document": asdict(info),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if directory.exists():
            shutil.rmtree(directory)
        temporary.rename(directory)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return vectorstore, info


def load_index(check: IndexCheck, embeddings: Embeddings) -> FAISS:
    """Load a saved index, confirming that it holds the expected number of chunks."""
    try:
        # FAISS stores its text in a pickle file, and loading a pickle can run code. This
        # is safe only for indexes this application wrote: never load one a user supplies.
        vectorstore = FAISS.load_local(
            str(check.directory),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        expected = check.manifest["document"]["chunks"]
    except Exception as exc:
        raise IndexLoadError("The saved index couldn't be read.") from exc
    if vectorstore.index.ntotal != expected:
        raise IndexLoadError("The saved index is incomplete.")
    return vectorstore


# --- Retrieval and generation -------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    document: str
    text: str
    page: int | None
    distance: float


def _page_number(metadata: dict[str, Any]) -> int | None:
    """Return the 1-based PDF page, which can differ from the printed number."""
    page = metadata.get("page")
    return int(page) + 1 if page is not None else None


def is_not_found(answer: str) -> bool:
    """True when the whole answer is the prompt's refusal sentence."""
    normalized = answer.replace("’", "'").strip().strip('"').casefold()
    return normalized == NOT_FOUND_MESSAGE.casefold()


class RagPipeline:
    def __init__(self, vectorstore: FAISS, document: DocumentInfo, llm: Any = None):
        self.vectorstore = vectorstore
        self.document = document
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Created on first use, so opening an index never requires an API key."""
        if self._llm is None:
            self._llm = _make_llm()
        return self._llm

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Run MMR retrieval, keeping the distance the plain retriever discards."""
        embedding = self.vectorstore.embeddings.embed_query(question)
        results = self.vectorstore.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=SEARCH_K,
            fetch_k=SEARCH_FETCH_K,
            lambda_mult=SEARCH_LAMBDA,
        )
        return [
            RetrievedChunk(
                rank=rank,
                document=self.document.name,
                text=doc.page_content,
                page=_page_number(doc.metadata),
                distance=float(score),
            )
            for rank, (doc, score) in enumerate(results, start=1)
        ]

    def _messages(self, question: str, chunks: list[RetrievedChunk]) -> Any:
        context = "\n\n".join(chunk.text for chunk in chunks)
        return _prompt().invoke({"context": context, "question": question})

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        return self.llm.invoke(self._messages(question, chunks)).content

    def stream(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        for part in self.llm.stream(self._messages(question, chunks)):
            if part.content:
                yield part.content

    def answer(self, question: str) -> str:
        return self.generate(question, self.retrieve(question))


def open_saved_pipeline(
    document: str | Path | None = None,
    *,
    on_stage: StageCallback = _no_op,
    embeddings: Embeddings | None = None,
    llm: Any = None,
) -> RagPipeline:
    """Open the document's saved index. Raises IndexNotAvailable rather than building one."""
    path = resolve_document(document)
    on_stage("checking", {})
    check = check_index(path)
    if check.status != "valid":
        raise IndexNotAvailable(check.reason)

    info = _info_from_manifest(path, check.manifest)
    on_stage("loading_index", {"chunks": info.chunks, "pages": info.pages_indexed})
    try:
        vectorstore = load_index(check, embeddings or _make_embeddings())
    except IndexLoadError as exc:
        raise IndexNotAvailable(str(exc)) from exc
    logger.info("Loaded the saved index for %s", path.name)
    return RagPipeline(vectorstore, info, llm)


def rebuild_index(
    document: str | Path | None = None,
    *,
    on_stage: StageCallback = _no_op,
    embeddings: Embeddings | None = None,
) -> DocumentInfo:
    """Explicitly parse, chunk and embed a document, and save the resulting index."""
    _, info = build_index(resolve_document(document), embeddings or _make_embeddings(), on_stage)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Show or rebuild the saved document index.")
    parser.add_argument("--document", help="PDF path. Defaults to RAG_DOCUMENT_PATH, then the default document.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index even if it's valid.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = resolve_document(args.document)
    try:
        check = check_index(path)
        print(f"Document: {path.name}\nIndex: {check.status}" + (f" ({check.reason})" if check.reason else ""))
        if args.rebuild:
            info = rebuild_index(path, on_stage=lambda stage, details: print(f"  {stage} {details or ''}"))
            print(f"Built: {info.chunks} chunks from {info.pages_indexed} of {info.pages_total} pages, "
                  f"skipped {info.pages_skipped}")
    except DocumentError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
