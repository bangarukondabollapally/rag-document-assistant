"""The RAG pipeline, independent of any UI and of how documents are stored.

    PDF → load → select content pages → chunk → embed → saved FAISS index
                                                              ↓
                  answer ← LLM ← context ← merged MMR retrieval ← selected indexes

Building an index and loading one are separate operations. document_store.py decides
when each happens, and application startup only ever loads.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

logger = logging.getLogger("rag_core")

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


# --- Document loading ---------------------------------------------------------


def load_document(path: Path, name: str | None = None) -> list[Document]:
    """Read a PDF into one Document per page, keeping page numbers in the metadata."""
    name = name or path.name
    if not path.is_file():
        raise DocumentError(f"The stored PDF for {name} is missing. Delete it and upload it again.")
    from langchain_community.document_loaders import PyPDFLoader

    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:
        raise DocumentError(f"{name} could not be read as a PDF.") from exc
    for page in pages:
        page.metadata["source"] = name
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
class IndexStats:
    title: str | None
    pages_total: int
    pages_indexed: int
    chunks: int
    pages_skipped: dict[str, int] = field(default_factory=dict)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_settings() -> dict[str, Any]:
    """Everything that changes an index's contents. A difference requires a rebuild."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "ingest_version": INGEST_VERSION,
    }


def _make_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _make_llm() -> Any:
    # Checked here because a blank key otherwise surfaces as a misleading connection error.
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise ValueError("GROQ_API_KEY is not set.")

    from langchain_groq import ChatGroq

    return ChatGroq(model=llm_model_name())


def build_index(
    document: Path,
    directory: Path,
    embeddings: Embeddings,
    *,
    name: str | None = None,
    on_stage: StageCallback = _no_op,
) -> tuple[FAISS, IndexStats]:
    """Create and save an index. Replaces any earlier index only after a full build."""
    name = name or document.name
    on_stage("loading", {})
    pages = load_document(document, name)
    kept, skipped = select_content_pages(pages)
    chunks = chunk_pages(kept)
    if not chunks:
        raise DocumentError(f"{name} has no extractable text. Scanned PDFs aren't supported yet.")

    title = pages[0].metadata.get("title") or None
    stats = IndexStats(title, len(pages), len(kept), len(chunks), skipped)
    logger.info("Indexing %s: kept %d of %d pages, skipped %s", name, len(kept), len(pages), skipped)
    on_stage("embedding", {"chunks": len(chunks), "pages": len(kept)})

    vectorstore = FAISS.from_documents(chunks, embeddings)

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".building-"))
    try:
        vectorstore.save_local(str(temporary))
        if directory.exists():
            shutil.rmtree(directory)
        temporary.rename(directory)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return vectorstore, stats


def load_index(directory: Path, embeddings: Embeddings, expected_chunks: int) -> FAISS:
    """Load a saved index, confirming that it holds the expected number of chunks."""
    try:
        # FAISS stores its text in a pickle file, and loading a pickle can run code. This
        # is safe only for indexes this application wrote: never load one a user supplies.
        vectorstore = FAISS.load_local(
            str(directory),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        raise IndexLoadError("The saved index couldn't be read.") from exc
    if vectorstore.index.ntotal != expected_chunks:
        raise IndexLoadError("The saved index is incomplete.")
    return vectorstore


# --- Retrieval and generation -------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    document_id: str
    document: str
    text: str
    page: int | None
    distance: float


@dataclass(frozen=True)
class SearchableDocument:
    id: str
    name: str
    index: FAISS


def _page_number(metadata: dict[str, Any]) -> int | None:
    """Return the 1-based PDF page, which can differ from the printed number."""
    page = metadata.get("page")
    return int(page) + 1 if page is not None else None


def is_not_found(answer: str) -> bool:
    """True when the whole answer is the prompt's refusal sentence."""
    normalized = answer.replace("’", "'").strip().strip('"').casefold()
    return normalized == NOT_FOUND_MESSAGE.casefold()


def retrieve(documents: Sequence[SearchableDocument], question: str) -> list[RetrievedChunk]:
    """Run MMR over the closest chunks of the given documents, and no others.

    Each index gives its nearest candidates. They're merged by distance, so the MMR step
    sees the same number of candidates as for a single document. For one document this
    is the same calculation as FAISS.max_marginal_relevance_search_with_score_by_vector.
    """
    if not documents:
        raise ValueError("retrieve needs at least one document.")
    import numpy as np
    from langchain_community.vectorstores.utils import maximal_marginal_relevance

    embedding = documents[0].index.embeddings.embed_query(question)
    query = np.array([embedding], dtype=np.float32)

    candidates = []
    for document in documents:
        distances, positions = document.index.index.search(query, SEARCH_FETCH_K)
        for distance, position in zip(distances[0], positions[0]):
            if position != -1:
                candidates.append((float(distance), document, int(position)))
    candidates.sort(key=lambda candidate: candidate[0])
    candidates = candidates[:SEARCH_FETCH_K]

    vectors = [document.index.index.reconstruct(position) for _, document, position in candidates]
    selected = maximal_marginal_relevance(query, vectors, lambda_mult=SEARCH_LAMBDA, k=SEARCH_K)

    chunks = []
    for rank, candidate_number in enumerate(selected, start=1):
        distance, document, position = candidates[candidate_number]
        stored = document.index.docstore.search(document.index.index_to_docstore_id[position])
        chunks.append(
            RetrievedChunk(
                rank=rank,
                document_id=document.id,
                document=document.name,
                text=stored.page_content,
                page=_page_number(stored.metadata),
                distance=distance,
            )
        )
    return chunks


class RagPipeline:
    def __init__(self, llm: Any = None):
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Created on first use, so opening an index never requires an API key."""
        if self._llm is None:
            self._llm = _make_llm()
        return self._llm

    def retrieve(self, documents: Sequence[SearchableDocument], question: str) -> list[RetrievedChunk]:
        return retrieve(documents, question)

    def _messages(self, question: str, chunks: list[RetrievedChunk]) -> Any:
        context = "\n\n".join(chunk.text for chunk in chunks)
        return _prompt().invoke({"context": context, "question": question})

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        return self.llm.invoke(self._messages(question, chunks)).content

    def stream(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        for part in self.llm.stream(self._messages(question, chunks)):
            if part.content:
                yield part.content

    def answer(self, documents: Sequence[SearchableDocument], question: str) -> str:
        return self.generate(question, self.retrieve(documents, question))
