"""RAG core tests. They use generated PDFs and fake embeddings, so they need no model or network."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import rag_core
from samples.make_sample import write_pdf


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Counts document embedding calls, which is the expensive step being avoided."""

    calls: int = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def make_embeddings():
    return CountingEmbeddings(size=16)


LLM = object()

PROSE = "\n".join(f"Sentence number {i} explains how neural networks learn from data." for i in range(12))
TOC = "\n".join(f"Section Title {i} . . . . . . {i * 7}" for i in range(1, 15))
TOC_NO_DOTS = "\n".join(f"Understanding Topic {i} {i * 9}" for i in range(1, 15))
BACK_INDEX = "\n".join(f"term {i}, {i * 3}, {i * 5}" for i in range(1, 40))
COPYRIGHT = "978-1-492-08218-7\nCopyright 2022 A Publisher.\nAll rights reserved.\nPrinted somewhere."
NUMERIC_SLIDE = "\n".join("1 1 2 4 5 6 7 8" for _ in range(12))
CODE = "\n".join(f"x{i} = model.layer({i})" for i in range(14))
SLIDE = "• Input Gate\n• The input gate is used to quantify new information."


def page(text, number=0):
    return Document(page_content=text, metadata={"page": number})


class ClassifyPageTests(unittest.TestCase):
    def test_removes_only_confident_non_content(self):
        cases = {
            "": "blank",
            " \n \n": "blank",
            TOC: "table_of_contents",
            TOC_NO_DOTS: "table_of_contents",
            BACK_INDEX: "back_of_book_index",
            COPYRIGHT: "copyright",
        }
        for text, reason in cases.items():
            self.assertEqual(rag_core.classify_page(text), reason, text[:40])

    def test_keeps_real_content(self):
        for text in (PROSE, NUMERIC_SLIDE, CODE, SLIDE, "Chapter 1"):
            self.assertIsNone(rag_core.classify_page(text), text[:40])

    def test_mentioning_copyright_alone_does_not_remove_a_page(self):
        self.assertIsNone(rag_core.classify_page("Fair use and copyright law. All rights reserved is a phrase."))

    def test_a_short_list_ending_in_numbers_is_kept(self):
        self.assertIsNone(rag_core.classify_page("Epoch one 10\nEpoch two 20\nEpoch three 30"))


class PageSelectionTests(unittest.TestCase):
    """Regression tests for the removed pages[20:-10] slice."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_content_at_both_ends_of_a_long_document_is_preserved(self):
        texts = ["FIRST-PAGE matrix multiplication"] + [PROSE] * 40 + ["LAST-PAGE output gate"]
        path = self.directory / "long.pdf"
        write_pdf(path, texts)

        kept, skipped = rag_core.select_content_pages(rag_core.load_document(path))
        chunks = rag_core.chunk_pages(kept)

        self.assertEqual(len(kept), 42)
        self.assertEqual(skipped, {})
        self.assertIn("FIRST-PAGE", chunks[0].page_content)
        self.assertIn("LAST-PAGE", chunks[-1].page_content)
        self.assertEqual(rag_core._page_number(chunks[-1].metadata), 42)

    def test_a_short_document_keeps_every_page(self):
        for count in (1, 5, 30):
            pages = [page(f"Content page {i}", i) for i in range(count)]
            kept, _ = rag_core.select_content_pages(pages)
            self.assertEqual(len(kept), count)

    def test_non_content_is_removed_wherever_it_appears_and_counted(self):
        pages = [page(t, i) for i, t in enumerate([COPYRIGHT, PROSE, "", BACK_INDEX, PROSE, TOC])]
        kept, skipped = rag_core.select_content_pages(pages)
        self.assertEqual([p.metadata["page"] for p in kept], [1, 4])
        self.assertEqual(skipped, {"copyright": 1, "blank": 1, "back_of_book_index": 1, "table_of_contents": 1})

    def test_chunk_page_numbers_refer_to_the_original_document(self):
        kept, _ = rag_core.select_content_pages([page("", 0), page("", 1), page(SLIDE, 2)])
        self.assertEqual(rag_core._page_number(rag_core.chunk_pages(kept)[0].metadata), 3)


class DocumentLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_reads_one_document_per_page_with_page_numbers(self):
        path = self.directory / "doc.pdf"
        write_pdf(path, ["Alpha", "Beta"], title="A Title")
        pages = rag_core.load_document(path)
        self.assertEqual([p.page_content.strip() for p in pages], ["Alpha", "Beta"])
        self.assertEqual([p.metadata["page"] for p in pages], [0, 1])
        self.assertEqual(pages[0].metadata["title"], "A Title")

    def test_stored_source_is_a_name_not_a_local_path(self):
        path = self.directory / "doc.pdf"
        write_pdf(path, ["Alpha"])
        self.assertEqual(rag_core.load_document(path)[0].metadata["source"], "doc.pdf")

    def test_missing_file_has_an_actionable_message(self):
        with self.assertRaisesRegex(rag_core.DocumentError, "RAG_DOCUMENT_PATH"):
            rag_core.load_document(self.directory / "absent.pdf")

    def test_invalid_pdf_is_reported(self):
        path = self.directory / "bad.pdf"
        path.write_bytes(b"this is not a pdf")
        with self.assertRaisesRegex(rag_core.DocumentError, "could not be read"):
            rag_core.load_document(path)


class DocumentResolutionTests(unittest.TestCase):
    def test_argument_then_environment_then_defaults(self):
        with mock.patch.dict(os.environ, {"RAG_DOCUMENT_PATH": "from/env.pdf"}):
            explicit = Path(tempfile.gettempdir()) / "explicit.pdf"
            self.assertEqual(rag_core.resolve_document(explicit), explicit)
            self.assertEqual(rag_core.resolve_document(), rag_core.ROOT / "from/env.pdf")

    def test_falls_back_to_the_sample_without_a_local_document(self):
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("RAG_DOCUMENT_PATH", None)
            with mock.patch.object(rag_core, "DEV_DOCUMENT", Path("/absent/dev.pdf")):
                self.assertEqual(rag_core.resolve_document(), rag_core.SAMPLE_DOCUMENT)

    def test_committed_sample_keeps_its_exact_bytes_after_checkout(self):
        # Converted line endings would invalidate the offsets inside the PDF.
        self.assertNotIn(b"\r\n", rag_core.SAMPLE_DOCUMENT.read_bytes())

    def test_committed_sample_is_a_usable_document(self):
        pages = rag_core.load_document(rag_core.SAMPLE_DOCUMENT)
        kept, skipped = rag_core.select_content_pages(pages)
        self.assertEqual(len(kept), len(pages))
        self.assertEqual(skipped, {})
        self.assertTrue(rag_core.suggested_questions(rag_core.SAMPLE_DOCUMENT))


class SuggestedQuestionsTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.document = self.directory / "doc.pdf"
        self.sidecar = self.directory / "doc.questions.json"

    def test_absent_file_gives_no_suggestions(self):
        self.assertEqual(rag_core.suggested_questions(self.document), [])

    def test_reads_questions_and_notes(self):
        self.sidecar.write_text(json.dumps([{"question": "Why?", "note": "A note"}, {"question": "How?"}]))
        self.assertEqual(
            rag_core.suggested_questions(self.document),
            [{"question": "Why?", "note": "A note"}, {"question": "How?", "note": None}],
        )

    def test_invalid_file_gives_no_suggestions(self):
        for content in ("not json", '{"question": "x"}', '[{"note": "no question"}]', "[1]"):
            self.sidecar.write_text(content)
            with self.assertLogs("rag_core", "WARNING"):
                self.assertEqual(rag_core.suggested_questions(self.document), [])


class PersistentIndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        patcher = mock.patch.dict(os.environ, {"RAG_INDEX_DIR": str(self.directory / "index")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.document = self.directory / "doc.pdf"
        write_pdf(self.document, [f"Topic {i} covers learning method number {i}." for i in range(10)], title="T")

    def open(self, document=None, **kwargs):
        embeddings = kwargs.pop("embeddings", None) or make_embeddings()
        pipeline = rag_core.open_saved_pipeline(document or self.document, embeddings=embeddings, llm=LLM, **kwargs)
        return pipeline, embeddings

    def rebuild(self, **kwargs):
        embeddings = make_embeddings()
        return rag_core.rebuild_index(self.document, embeddings=embeddings, **kwargs), embeddings

    def index_dir(self):
        return rag_core.check_index(self.document).directory

    def assert_not_available(self, reason):
        embeddings = make_embeddings()
        with self.assertRaisesRegex(rag_core.IndexNotAvailable, reason):
            rag_core.open_saved_pipeline(self.document, embeddings=embeddings, llm=LLM)
        self.assertEqual(embeddings.calls, 0)

    def test_open_without_an_index_raises_and_builds_nothing(self):
        self.assert_not_available("No saved index")
        self.assertEqual(rag_core.check_index(self.document).status, "missing")
        self.assertFalse((self.directory / "index").exists())

    def test_rebuild_saves_a_valid_index_once(self):
        info, embeddings = self.rebuild()
        self.assertEqual(embeddings.calls, 1)
        self.assertEqual(info.pages_total, 10)
        self.assertEqual(rag_core.check_index(self.document).status, "valid")

    def test_open_after_rebuild_loads_without_re_embedding(self):
        self.rebuild()
        pipeline, embeddings = self.open()
        self.assertEqual(embeddings.calls, 0)
        self.assertEqual(pipeline.document.pages_total, 10)

    def test_reopened_index_retrieves_identically(self):
        self.rebuild()
        first, _ = self.open()
        second, _ = self.open()
        for question in ("Topic 3", "learning method", "something else"):
            self.assertEqual(first.retrieve(question), second.retrieve(question))

    def test_retrieval_includes_page_and_document(self):
        self.rebuild()
        chunk = self.open()[0].retrieve("Topic 3")[0]
        self.assertEqual(chunk.document, "doc.pdf")
        self.assertIn(chunk.page, range(1, 11))

    def test_changed_document_is_not_indexed_until_rebuilt(self):
        self.rebuild()
        write_pdf(self.document, ["Completely different content."])
        self.assert_not_available("No saved index")
        self.rebuild()
        self.assertEqual(self.open()[0].document.pages_total, 1)

    def test_renamed_document_reuses_its_index_under_the_new_name(self):
        self.rebuild()
        renamed = self.directory / "renamed.pdf"
        self.document.rename(renamed)
        pipeline, embeddings = self.open(renamed)
        self.assertEqual(embeddings.calls, 0)
        self.assertEqual(pipeline.document.name, "renamed.pdf")

    def test_changed_settings_make_the_index_unavailable_until_rebuilt(self):
        for attribute, value in (
            ("CHUNK_SIZE", 400),
            ("CHUNK_OVERLAP", 50),
            ("EMBEDDING_MODEL", "other/model"),
            ("INGEST_VERSION", 99),
        ):
            with self.subTest(attribute):
                self.rebuild()
                with mock.patch.object(rag_core, attribute, value):
                    self.assert_not_available("settings")
                self.open()

    def test_damaged_index_files_are_reported_not_rebuilt(self):
        for name in ("index.faiss", "index.pkl"):
            with self.subTest(name):
                self.rebuild()
                (self.index_dir() / name).write_bytes(b"damaged")
                self.assert_not_available("couldn't be read")

    def test_damaged_manifest_is_reported(self):
        self.rebuild()
        (self.index_dir() / "manifest.json").write_text("not json")
        self.assertEqual(rag_core.check_index(self.document).status, "stale")
        self.assert_not_available("manifest")

    def test_incomplete_index_is_reported(self):
        self.rebuild()
        manifest_path = self.index_dir() / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["document"]["chunks"] += 1
        manifest_path.write_text(json.dumps(manifest))
        self.assert_not_available("incomplete")

    def test_interrupted_build_leaves_no_valid_index_or_temporary_files(self):
        with mock.patch.object(rag_core.FAISS, "save_local", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.rebuild()
        self.assertEqual(rag_core.check_index(self.document).status, "missing")
        self.assertEqual(list((self.directory / "index").iterdir()), [])

    def test_failed_rebuild_keeps_the_existing_index(self):
        self.rebuild()
        with mock.patch.object(rag_core.FAISS, "save_local", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.rebuild()
        self.assertEqual(rag_core.check_index(self.document).status, "valid")
        self.open()

    def test_document_without_text_fails_clearly_and_saves_nothing(self):
        write_pdf(self.document, ["", ""])
        with self.assertRaisesRegex(rag_core.DocumentError, "no extractable text"):
            self.rebuild()
        self.assertEqual(rag_core.check_index(self.document).status, "missing")

    def test_missing_document_is_reported(self):
        self.document.unlink()
        with self.assertRaises(rag_core.DocumentError):
            self.open()

    def test_opening_does_not_create_the_llm_until_it_is_used(self):
        self.rebuild()
        with mock.patch.object(rag_core, "_make_llm", return_value="the llm") as make_llm:
            pipeline = rag_core.open_saved_pipeline(self.document, embeddings=make_embeddings())
            make_llm.assert_not_called()
            self.assertEqual(pipeline.llm, "the llm")
            self.assertEqual(pipeline.llm, "the llm")
            make_llm.assert_called_once()

    def test_stages_show_which_operation_ran(self):
        seen = []
        self.rebuild(on_stage=lambda stage, details: seen.append(stage))
        self.assertEqual(seen, ["loading", "embedding"])
        seen.clear()
        self.open(on_stage=lambda stage, details: seen.append(stage))
        self.assertEqual(seen, ["checking", "loading_index"])


class PromptTests(unittest.TestCase):
    def test_prompt_names_no_particular_document(self):
        system = rag_core._prompt().messages[0].prompt.template
        self.assertNotIn("textbook", system)
        self.assertIn(rag_core.NOT_FOUND_MESSAGE, system)

    def test_retrieval_settings_are_unchanged(self):
        self.assertEqual(
            (rag_core.CHUNK_SIZE, rag_core.CHUNK_OVERLAP, rag_core.SEARCH_K, rag_core.SEARCH_FETCH_K, rag_core.SEARCH_LAMBDA),
            (500, 100, 6, 20, 0.7),
        )
        self.assertEqual(rag_core.EMBEDDING_MODEL, "sentence-transformers/all-mpnet-base-v2")


if __name__ == "__main__":
    unittest.main()
