"""Shared test helpers. Nothing here needs a real model, the network or an API key."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from samples.make_sample import write_pdf

KEYWORDS = ("alpha", "beta", "gamma", "delta")


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Counts document embedding calls, which is the expensive step being avoided."""

    calls: int = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


class KeywordEmbeddings(Embeddings):
    """Puts texts sharing a keyword close together, so retrieval tests are predictable."""

    def __init__(self):
        self.document_calls = 0

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        counts = [float(lowered.count(word)) for word in KEYWORDS]
        # A small unique component, so that different chunks are never identical.
        counts.append(sum(ord(character) for character in lowered) % 97 / 970)
        size = sum(value * value for value in counts) ** 0.5 or 1.0
        return [value / size for value in counts]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeLLM:
    def __init__(self, answer: str = "A generated answer."):
        self.answer = answer

    def invoke(self, messages):
        return SimpleNamespace(content=self.answer)

    def stream(self, messages):
        for word in self.answer.split(" "):
            yield SimpleNamespace(content=word + " ")


def pdf_bytes(pages: list[str], title: str | None = None) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "document.pdf"
        write_pdf(path, pages, title=title)
        return path.read_bytes()


def topic_pages(keyword: str, count: int = 4) -> list[str]:
    return [f"Section {number} explains {keyword} in {keyword} systems, part {number}." for number in range(1, count + 1)]
