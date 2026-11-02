"""Writes samples/rag-guide.pdf, an original document that's safe to commit and share.

Run from the project root: python samples/make_sample.py
"""

from __future__ import annotations

from pathlib import Path

GUIDE_PAGES = [
    """How Retrieval-Augmented Generation Works

Retrieval-augmented generation, or RAG, answers questions using a set of documents.
A language model on its own answers from what it learned during training, so it can
be out of date and knows nothing about your private files. RAG adds a search step:
it finds passages related to the question and gives them to the model as context.
The model is then asked to answer from those passages.

The approach has two phases. Indexing prepares the documents once, ahead of time.
Querying runs each time someone asks a question.""",
    """Indexing: Preparing Documents

Indexing begins by extracting the text of each document. Long text is then split into
smaller chunks, because a search works best when each result covers one idea.
Overlap between neighbouring chunks keeps a sentence that crosses a boundary whole.

An embedding model converts each chunk into a vector, a list of numbers that captures
its meaning. Chunks with similar meanings get vectors that are close together. The
vectors are stored in a vector index, which can find the nearest ones quickly. Saving
the index means the slow embedding work doesn't have to be repeated at every start.""",
    """Querying: Finding Relevant Passages

To answer a question, the system embeds the question with the same embedding model and
searches the index for the nearest chunks. Plain similarity search can return several
chunks that say nearly the same thing.

Maximal marginal relevance, or MMR, reduces that repetition. It first fetches a larger
set of candidates, then picks results one at a time, balancing closeness to the question
against similarity to the results it has already picked. A lambda value of 1 considers
only closeness to the question, and lower values give more weight to variety.""",
    """Generating and Checking the Answer

The selected passages and the question are combined into a prompt, which instructs the
model to answer from the passages and to say so when they don't contain the answer.
Grounding like this reduces invented answers but doesn't eliminate them.

Showing the retrieved passages beside the answer lets a reader check the answer against
its sources. A passage's distance score shows how similar it is to the question. It
doesn't show that the passage is correct or that the answer used it faithfully. Reading
the source remains the reliable way to verify a claim.""",
]


def _escape(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return escaped.encode("latin-1", "replace")


def write_pdf(path: Path, pages: list[str], title: str | None = None) -> None:
    """Write a minimal text PDF using a standard font, so no PDF library is needed."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    page_ids = [4 + index * 2 for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    for page_id, text in zip(page_ids, pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        commands = [b"BT /F1 10 Tf 13 TL 50 740 Td"]
        for line in text.splitlines():
            commands.append(b"(" + _escape(line) + b") Tj T*")
        commands.append(b"ET")
        stream = b"\n".join(commands)
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    if title:
        objects.append(b"<< /Title (" + _escape(title) + b") >>")

    # The second line is the PDF convention marking the file as binary, which also
    # stops git changing its line endings and invalidating the offsets below.
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        output += b"%010d 00000 n \n" % offset
    info = f" /Info {len(objects)} 0 R" if title else ""
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R{info} >>\nstartxref\n{xref}\n%%EOF\n"
    output += trailer.encode()
    path.write_bytes(bytes(output))


if __name__ == "__main__":
    write_pdf(
        Path(__file__).resolve().parent / "rag-guide.pdf",
        GUIDE_PAGES,
        title="How Retrieval-Augmented Generation Works",
    )
