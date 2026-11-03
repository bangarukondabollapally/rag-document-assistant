"""RAG core tests. They use generated PDFs and fake embeddings, so they need no model or network."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.documents import Document

import rag_core
from rag_core import SearchableDocument
from samples.make_sample import write_pdf
from tests.support import SAMPLE_DOCUMENT, CountingEmbeddings, KeywordEmbeddings, topic_pages


def make_embeddings():
    return CountingEmbeddings(size=16)


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

    def test_errors_and_metadata_use_the_given_display_name(self):
        path = self.directory / "original.pdf"
        write_pdf(path, ["Alpha"])
        self.assertEqual(rag_core.load_document(path, "Report.pdf")[0].metadata["source"], "Report.pdf")
        path.write_bytes(b"not a pdf")
        with self.assertRaisesRegex(rag_core.DocumentError, "Report.pdf"):
            rag_core.load_document(path, "Report.pdf")

    def test_missing_file_has_an_actionable_message(self):
        with self.assertRaisesRegex(rag_core.DocumentError, "missing"):
            rag_core.load_document(self.directory / "absent.pdf")

    def test_invalid_pdf_is_reported(self):
        path = self.directory / "bad.pdf"
        path.write_bytes(b"this is not a pdf")
        with self.assertRaisesRegex(rag_core.DocumentError, "could not be read"):
            rag_core.load_document(path)


class SampleDocumentTests(unittest.TestCase):
    def test_committed_sample_keeps_its_exact_bytes_after_checkout(self):
        # Converted line endings would invalidate the offsets inside the PDF.
        self.assertNotIn(b"\r\n", SAMPLE_DOCUMENT.read_bytes())

    def test_committed_sample_is_a_usable_document(self):
        pages = rag_core.load_document(SAMPLE_DOCUMENT)
        kept, skipped = rag_core.select_content_pages(pages)
        self.assertEqual(len(kept), len(pages))
        self.assertEqual(skipped, {})


class IndexBuildAndLoadTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.document = self.directory / "doc.pdf"
        write_pdf(self.document, [f"Topic {i} covers learning method number {i}." for i in range(10)], title="T")
        self.index = self.directory / "doc" / "index"

    def build(self, embeddings=None, **kwargs):
        return rag_core.build_index(self.document, self.index, embeddings or make_embeddings(), **kwargs)

    def test_build_saves_an_index_and_reports_its_contents(self):
        embeddings = make_embeddings()
        _, stats = self.build(embeddings)
        self.assertEqual(embeddings.calls, 1)
        self.assertEqual((stats.title, stats.pages_total, stats.pages_indexed), ("T", 10, 10))
        self.assertTrue((self.index / "index.faiss").is_file())
        self.assertTrue((self.index / "index.pkl").is_file())

    def test_load_does_not_embed_anything(self):
        _, stats = self.build()
        embeddings = make_embeddings()
        loaded = rag_core.load_index(self.index, embeddings, stats.chunks)
        self.assertEqual(loaded.index.ntotal, stats.chunks)
        self.assertEqual(embeddings.calls, 0)

    def test_a_loaded_index_retrieves_exactly_what_the_built_one_did(self):
        embeddings = KeywordEmbeddings()
        built, stats = self.build(embeddings)
        loaded = rag_core.load_index(self.index, embeddings, stats.chunks)
        for question in ("Topic 3", "learning method", "something else"):
            self.assertEqual(
                rag_core.retrieve([SearchableDocument("a", "doc.pdf", built)], question),
                rag_core.retrieve([SearchableDocument("a", "doc.pdf", loaded)], question),
            )

    def test_damaged_index_files_are_reported(self):
        for name in ("index.faiss", "index.pkl"):
            with self.subTest(name):
                _, stats = self.build()
                (self.index / name).write_bytes(b"damaged")
                with self.assertRaisesRegex(rag_core.IndexLoadError, "couldn't be read"):
                    rag_core.load_index(self.index, make_embeddings(), stats.chunks)

    def test_an_incomplete_index_is_reported(self):
        _, stats = self.build()
        with self.assertRaisesRegex(rag_core.IndexLoadError, "incomplete"):
            rag_core.load_index(self.index, make_embeddings(), stats.chunks + 1)

    def test_a_missing_index_is_reported(self):
        with self.assertRaises(rag_core.IndexLoadError):
            rag_core.load_index(self.index, make_embeddings(), 1)

    def test_an_interrupted_build_leaves_no_index_or_temporary_files(self):
        with mock.patch.object(rag_core.FAISS, "save_local", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.build()
        self.assertFalse(self.index.exists())
        self.assertEqual(list(self.index.parent.iterdir()), [])

    def test_a_failed_rebuild_keeps_the_existing_index(self):
        _, stats = self.build()
        with mock.patch.object(rag_core.FAISS, "save_local", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.build()
        self.assertIsNotNone(rag_core.load_index(self.index, make_embeddings(), stats.chunks))

    def test_a_document_without_text_fails_clearly_and_saves_nothing(self):
        write_pdf(self.document, ["", ""])
        with self.assertRaisesRegex(rag_core.DocumentError, "no extractable text"):
            self.build()
        self.assertFalse(self.index.exists())

    def test_stages_are_reported_in_order(self):
        seen = []
        self.build(on_stage=lambda stage, details: seen.append(stage))
        self.assertEqual(seen, ["loading", "embedding"])


class ScopedRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.embeddings = KeywordEmbeddings()

    def document(self, keyword, count=4):
        path = self.directory / f"{keyword}.pdf"
        write_pdf(path, topic_pages(keyword, count))
        vectorstore, _ = rag_core.build_index(path, self.directory / keyword, self.embeddings)
        return SearchableDocument(f"id-{keyword}", f"{keyword}.pdf", vectorstore)

    def test_one_document_gives_exactly_the_native_faiss_mmr_result(self):
        document = self.document("alpha", 30)
        for question in ("alpha", "part 3", "beta", "alpha systems part 12"):
            native = document.index.max_marginal_relevance_search_with_score_by_vector(
                self.embeddings.embed_query(question),
                k=rag_core.SEARCH_K,
                fetch_k=rag_core.SEARCH_FETCH_K,
                lambda_mult=rag_core.SEARCH_LAMBDA,
            )
            merged = rag_core.retrieve([document], question)
            self.assertEqual([chunk.text for chunk in merged], [doc.page_content for doc, _ in native])
            self.assertEqual([chunk.distance for chunk in merged], [float(score) for _, score in native])

    def test_results_come_only_from_the_given_documents(self):
        alpha, beta, gamma = self.document("alpha"), self.document("beta"), self.document("gamma")
        chunks = rag_core.retrieve([beta], "alpha")
        self.assertEqual({chunk.document_id for chunk in chunks}, {"id-beta"})
        chunks = rag_core.retrieve([alpha, beta], "gamma")
        self.assertLessEqual({chunk.document_id for chunk in chunks}, {"id-alpha", "id-beta"})
        self.assertIsNotNone(gamma)

    def test_several_documents_can_all_contribute(self):
        chunks = rag_core.retrieve([self.document("alpha"), self.document("beta")], "alpha beta")
        self.assertEqual({chunk.document for chunk in chunks}, {"alpha.pdf", "beta.pdf"})

    def test_the_closest_document_supplies_the_top_result(self):
        chunks = rag_core.retrieve([self.document("alpha"), self.document("beta")], "beta")
        self.assertEqual(chunks[0].document, "beta.pdf")

    def test_the_total_stays_within_the_configured_limit(self):
        documents = [self.document(word, 12) for word in ("alpha", "beta", "gamma")]
        self.assertEqual(len(rag_core.retrieve(documents, "alpha beta gamma")), rag_core.SEARCH_K)

    def test_ranks_are_sequential_across_documents(self):
        chunks = rag_core.retrieve([self.document("alpha"), self.document("beta")], "alpha beta")
        self.assertEqual([chunk.rank for chunk in chunks], list(range(1, len(chunks) + 1)))

    def test_each_chunk_keeps_its_own_document_and_page(self):
        chunks = rag_core.retrieve([self.document("alpha"), self.document("beta")], "alpha beta")
        for chunk in chunks:
            self.assertIn(chunk.document.removesuffix(".pdf"), chunk.text)
            self.assertIn(f"part {chunk.page}", chunk.text)

    def test_a_document_smaller_than_the_candidate_count_works(self):
        self.assertEqual(len(rag_core.retrieve([self.document("alpha", 2)], "alpha")), 2)

    def test_searching_no_documents_is_an_error_not_a_global_search(self):
        with self.assertRaises(ValueError):
            rag_core.retrieve([], "alpha")


class PipelineTests(unittest.TestCase):
    def test_opening_does_not_create_the_llm_until_it_is_used(self):
        with mock.patch.object(rag_core, "_make_llm", return_value="the llm") as make_llm:
            pipeline = rag_core.RagPipeline()
            make_llm.assert_not_called()
            self.assertEqual(pipeline.llm, "the llm")
            self.assertEqual(pipeline.llm, "the llm")
            make_llm.assert_called_once()


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
