"""The document registry: uploads, indexing lifecycle, persistence and deletion."""

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import document_store
import rag_core
from document_store import DocumentBusy, DocumentNotFound, DocumentNotReady, DocumentStore, UploadRejected, safe_display_name
from samples.make_sample import write_pdf
from tests.support import CountingEmbeddings, KeywordEmbeddings, pdf_bytes, topic_pages


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.root = self.directory / "documents"
        self.embeddings = CountingEmbeddings(size=16)
        self.store = DocumentStore(self.root, embeddings=self.embeddings)
        self.store.load()

    def upload(self, pages=None, filename="paper.pdf", content_type="application/pdf"):
        body = pdf_bytes(pages or topic_pages("alpha"))
        return self.store.add_upload(io.BytesIO(body), filename, content_type)

    def reopened(self, embeddings=None):
        store = DocumentStore(self.root, embeddings=embeddings or CountingEmbeddings(size=16))
        store.load()
        return store

    def manifest(self, document_id):
        return json.loads((self.root / document_id / "manifest.json").read_text(encoding="utf-8"))

    def edit_manifest(self, document_id, **changes):
        path = self.root / document_id / "manifest.json"
        path.write_text(json.dumps({**self.manifest(document_id), **changes}), encoding="utf-8")


class RegistryTests(StoreTestCase):
    def test_a_new_store_has_no_documents(self):
        self.assertEqual(self.store.list(), [])

    def test_creating_a_document_records_its_metadata(self):
        record = self.upload(filename="My Paper.pdf")
        self.assertRegex(record.id, r"^[0-9a-f]{32}$")
        self.assertEqual(record.name, "My Paper.pdf")
        self.assertEqual(record.status, "uploaded")
        self.assertEqual(len(record.sha256), 64)
        self.assertGreater(record.size_bytes, 0)
        self.assertTrue(record.created_at)
        self.assertIsNone(record.error)

    def test_listing_returns_documents_oldest_first(self):
        first = self.upload(topic_pages("alpha"), "first.pdf")
        second = self.upload(topic_pages("beta"), "second.pdf")
        self.store.get(first.id).created_at = "2020-01-01"
        self.store.get(second.id).created_at = "2021-01-01"
        self.assertEqual([record.name for record in self.store.list()], ["first.pdf", "second.pdf"])

    def test_getting_a_document_returns_its_record(self):
        record = self.upload()
        self.assertIs(self.store.get(record.id), record)

    def test_getting_an_unknown_document_raises(self):
        with self.assertRaises(DocumentNotFound):
            self.store.get("0" * 32)

    def test_metadata_and_pdf_are_written_to_the_document_folder(self):
        record = self.upload()
        self.assertTrue((self.root / record.id / "original.pdf").is_file())
        self.assertEqual(self.manifest(record.id)["name"], "paper.pdf")
        self.assertEqual(self.manifest(record.id)["sha256"], record.sha256)

    def test_deleting_removes_the_pdf_metadata_and_index(self):
        record = self.upload()
        self.store.index_now(record.id)
        self.store.delete(record.id)
        self.assertFalse((self.root / record.id).exists())
        self.assertEqual(self.store.list(), [])
        with self.assertRaises(DocumentNotFound):
            self.store.get(record.id)
        self.assertEqual(self.reopened().list(), [])

    def test_deleting_one_document_leaves_the_others(self):
        keep = self.upload(topic_pages("alpha"), "keep.pdf")
        remove = self.upload(topic_pages("beta"), "remove.pdf")
        self.store.index_now(keep.id)
        self.store.index_now(remove.id)
        before = (self.root / keep.id / "index" / "index.faiss").read_bytes()
        self.store.delete(remove.id)
        self.assertEqual((self.root / keep.id / "index" / "index.faiss").read_bytes(), before)
        self.assertEqual(self.store.get(keep.id).status, "ready")

    def test_a_deleted_document_can_be_uploaded_again(self):
        record = self.upload()
        self.store.delete(record.id)
        self.assertNotEqual(self.upload().id, record.id)

    def test_a_document_being_indexed_cannot_be_deleted(self):
        record = self.upload()
        self.store._busy.add(record.id)
        with self.assertRaises(DocumentBusy):
            self.store.delete(record.id)
        self.assertTrue((self.root / record.id).exists())

    def test_deleting_an_unknown_document_raises(self):
        with self.assertRaises(DocumentNotFound):
            self.store.delete("0" * 32)


class UploadValidationTests(StoreTestCase):
    def assert_rejected(self, body, filename, content_type, status, code):
        with self.assertRaises(UploadRejected) as caught:
            self.store.add_upload(io.BytesIO(body), filename, content_type)
        self.assertEqual((caught.exception.status_code, caught.exception.code), (status, code))
        self.assertEqual(self.store.list(), [])
        leftovers = list(self.root.iterdir()) if self.root.exists() else []
        self.assertEqual(leftovers, [], "nothing should be left behind")

    def test_a_valid_pdf_is_accepted(self):
        self.assertEqual(self.upload().status, "uploaded")

    def test_a_wrong_extension_is_rejected(self):
        self.assert_rejected(pdf_bytes(["x"]), "notes.txt", "application/pdf", 415, "invalid_type")

    def test_a_wrong_content_type_is_rejected(self):
        self.assert_rejected(pdf_bytes(["x"]), "notes.pdf", "text/html", 415, "invalid_type")

    def test_content_that_is_not_a_pdf_is_rejected_despite_its_name(self):
        self.assert_rejected(b"<html>not a pdf</html>", "fake.pdf", "application/pdf", 415, "invalid_type")

    def test_an_empty_file_is_rejected(self):
        self.assert_rejected(b"", "empty.pdf", "application/pdf", 415, "invalid_type")

    def test_an_oversized_file_is_rejected_and_not_kept(self):
        body = b"%PDF-1.4\n" + b"x" * 5000
        with mock.patch.dict(os.environ, {"RAG_MAX_UPLOAD_MB": "0.001"}):
            self.assert_rejected(body, "big.pdf", "application/pdf", 413, "too_large")

    def test_a_file_exactly_at_the_limit_is_accepted(self):
        body = pdf_bytes(["x"])
        with mock.patch.dict(os.environ, {"RAG_MAX_UPLOAD_MB": str(len(body) / (1024 * 1024))}):
            self.assertEqual(self.store.add_upload(io.BytesIO(body), "ok.pdf", "application/pdf").status, "uploaded")

    def test_a_duplicate_is_rejected_and_names_the_original(self):
        original = self.upload(filename="original.pdf")
        with self.assertRaises(UploadRejected) as caught:
            self.upload(filename="renamed copy.pdf")
        self.assertEqual((caught.exception.status_code, caught.exception.code), (409, "duplicate"))
        self.assertEqual(caught.exception.document_id, original.id)
        self.assertIn("original.pdf", caught.exception.message)
        self.assertEqual(len(self.store.list()), 1)

    def test_a_duplicate_does_not_overwrite_the_existing_document(self):
        original = self.upload(filename="original.pdf")
        self.store.index_now(original.id)
        before = self.manifest(original.id)
        with self.assertRaises(UploadRejected):
            self.upload(filename="other.pdf")
        self.assertEqual(self.manifest(original.id), before)

    def test_same_name_with_different_content_is_a_separate_document(self):
        first = self.upload(topic_pages("alpha"), "same.pdf")
        second = self.upload(topic_pages("beta"), "same.pdf")
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(self.store.list()), 2)


class UnsafeFilenameTests(StoreTestCase):
    def test_path_traversal_names_are_reduced_to_their_last_part(self):
        for unsafe in ("../../evil.pdf", "..\\..\\evil.pdf", "/etc/evil.pdf", "C:\\Windows\\evil.pdf"):
            self.assertEqual(safe_display_name(unsafe), "evil.pdf", unsafe)

    def test_control_characters_are_removed(self):
        self.assertEqual(safe_display_name("a\x00b\r\nc.pdf"), "abc.pdf")

    def test_empty_and_dot_only_names_get_a_default(self):
        for name in (None, "", "   ", "..", "../"):
            self.assertEqual(safe_display_name(name), "document.pdf", repr(name))

    def test_very_long_names_are_shortened_and_stay_pdfs(self):
        name = safe_display_name("x" * 1000 + ".pdf")
        self.assertLessEqual(len(name), 200)
        self.assertTrue(name.endswith(".pdf"))

    def test_a_traversal_upload_writes_only_inside_its_own_folder(self):
        record = self.upload(filename="../../../outside.pdf")
        self.assertEqual(record.name, "outside.pdf")
        self.assertFalse((self.directory / "outside.pdf").exists())
        self.assertFalse((self.root.parent.parent / "outside.pdf").exists())
        self.assertEqual(sorted(p.name for p in (self.root / record.id).iterdir()), ["manifest.json", "original.pdf"])

    def test_the_client_filename_is_never_the_stored_filename(self):
        record = self.upload(filename="report.pdf")
        self.assertFalse((self.root / record.id / "report.pdf").exists())

    def test_ids_that_could_escape_the_folder_are_refused(self):
        for unsafe in ("../secret", "..", "a/b", "A" * 32, "x" * 32):
            with self.assertRaises(DocumentNotFound, msg=unsafe):
                self.store.pdf_path(unsafe)


class IndexingLifecycleTests(StoreTestCase):
    def test_indexing_success_records_page_and_chunk_information(self):
        record = self.store.index_now(self.upload(topic_pages("alpha", 3)).id)
        self.assertEqual(record.status, "ready")
        self.assertEqual((record.pages_total, record.pages_indexed), (3, 3))
        self.assertGreaterEqual(record.chunks, 3)
        self.assertEqual(record.settings, rag_core.index_settings())
        self.assertTrue(record.indexed_at)
        self.assertEqual(self.embeddings.calls, 1)

    def test_status_passes_through_indexing_while_the_build_runs(self):
        record = self.upload()
        seen = []
        self.store.index_now(record.id, on_stage=lambda stage, details: seen.append(self.manifest(record.id)["status"]))
        self.assertEqual(seen, ["indexing", "indexing"])
        self.assertEqual(self.manifest(record.id)["status"], "ready")

    def test_uploading_alone_does_not_index(self):
        self.upload()
        self.assertEqual(self.embeddings.calls, 0)

    def test_background_indexing_finishes_and_persists(self):
        record = self.upload()
        self.store.request_indexing(record.id)
        self.store.wait_until_idle()
        self.assertEqual(self.store.get(record.id).status, "ready")
        self.assertEqual(self.manifest(record.id)["status"], "ready")
        self.assertTrue((self.root / record.id / "index" / "index.faiss").is_file())

    def test_requesting_indexing_twice_while_busy_is_refused(self):
        record = self.upload()
        self.store._busy.add(record.id)
        with self.assertRaises(DocumentBusy):
            self.store.request_indexing(record.id)

    def test_an_unreadable_pdf_records_a_useful_failure(self):
        record = self.store.add_upload(io.BytesIO(b"%PDF-1.4 garbage"), "broken.pdf", "application/pdf")
        result = self.store.index_now(record.id)
        self.assertEqual(result.status, "failed")
        self.assertIn("broken.pdf", result.error)
        self.assertEqual(self.manifest(record.id)["status"], "failed")
        self.assertEqual(self.embeddings.calls, 0)

    def test_a_pdf_without_text_records_a_useful_failure(self):
        result = self.store.index_now(self.upload(["", ""], "scan.pdf").id)
        self.assertEqual(result.status, "failed")
        self.assertIn("no extractable text", result.error)

    def test_an_unexpected_failure_is_recorded_without_leaking_details(self):
        record = self.upload()
        with mock.patch.object(rag_core.FAISS, "from_documents", side_effect=RuntimeError("secret path C:/x")):
            result = self.store.index_now(record.id)
        self.assertEqual(result.status, "failed")
        self.assertNotIn("secret", result.error)

    def test_a_failed_background_job_does_not_stop_the_worker(self):
        broken = self.store.add_upload(io.BytesIO(b"%PDF-1.4 garbage"), "broken.pdf", "application/pdf")
        good = self.upload()
        self.store.request_indexing(broken.id)
        self.store.request_indexing(good.id)
        self.store.wait_until_idle()
        self.assertEqual(self.store.get(broken.id).status, "failed")
        self.assertEqual(self.store.get(good.id).status, "ready")

    def test_a_failed_document_can_be_retried(self):
        record = self.upload()
        with mock.patch.object(rag_core.FAISS, "from_documents", side_effect=RuntimeError("temporary")):
            self.store.index_now(record.id)
        self.assertEqual(self.store.get(record.id).status, "failed")
        result = self.store.index_now(record.id)
        self.assertEqual(result.status, "ready")
        self.assertIsNone(result.error)

    def test_an_interrupted_save_leaves_no_temporary_files(self):
        record = self.upload()
        with mock.patch.object(rag_core.FAISS, "save_local", side_effect=OSError("disk full")):
            self.store.index_now(record.id)
        self.assertEqual(sorted(p.name for p in (self.root / record.id).iterdir()), ["manifest.json", "original.pdf"])


class PersistenceTests(StoreTestCase):
    def indexed(self, keyword="alpha"):
        return self.store.index_now(self.upload(topic_pages(keyword), f"{keyword}.pdf").id)

    def test_an_index_survives_a_restart(self):
        record = self.indexed()
        restarted = self.reopened()
        self.assertEqual(restarted.get(record.id).status, "ready")
        self.assertEqual(restarted.get(record.id).chunks, record.chunks)

    def test_an_existing_index_loads_without_re_embedding(self):
        record = self.indexed()
        embeddings = CountingEmbeddings(size=16)
        restarted = self.reopened(embeddings)
        self.assertIsNotNone(restarted.open_index(record.id))
        self.assertEqual(embeddings.calls, 0)

    def test_an_index_is_opened_only_once(self):
        record = self.indexed()
        restarted = self.reopened()
        self.assertIs(restarted.open_index(record.id), restarted.open_index(record.id))

    def test_reindexing_explicitly_rebuilds_that_document_only(self):
        target = self.indexed("alpha")
        other = self.indexed("beta")
        other_files = {p.name: p.read_bytes() for p in (self.root / other.id / "index").iterdir()}
        other_manifest = self.manifest(other.id)
        before = self.embeddings.calls

        result = self.store.index_now(target.id)

        self.assertEqual(result.status, "ready")
        self.assertEqual(self.embeddings.calls, before + 1)
        self.assertEqual({p.name: p.read_bytes() for p in (self.root / other.id / "index").iterdir()}, other_files)
        self.assertEqual(self.manifest(other.id), other_manifest)

    def test_a_failed_reindex_keeps_the_previous_saved_index_on_disk(self):
        record = self.indexed()
        before = (self.root / record.id / "index" / "index.faiss").read_bytes()
        with mock.patch.object(rag_core.FAISS, "save_local", side_effect=OSError("disk full")):
            self.store.index_now(record.id)
        self.assertEqual((self.root / record.id / "index" / "index.faiss").read_bytes(), before)

    def test_interrupted_indexing_is_reported_and_not_restarted(self):
        record = self.upload()
        self.edit_manifest(record.id, status="indexing")
        restarted = self.reopened()
        self.assertEqual(restarted.get(record.id).status, "failed")
        self.assertIn("interrupted", restarted.get(record.id).error)
        self.assertFalse((self.root / record.id / "index").exists())

    def test_a_changed_pdf_is_reported_and_not_rebuilt(self):
        record = self.indexed()
        write_pdf(self.root / record.id / "original.pdf", ["Changed."])
        restarted = self.reopened()
        self.assertEqual(restarted.get(record.id).status, "failed")
        self.assertIn("changed", restarted.get(record.id).error)

    def test_changed_settings_are_reported_and_not_rebuilt(self):
        for attribute, value in (
            ("CHUNK_SIZE", 400),
            ("CHUNK_OVERLAP", 50),
            ("EMBEDDING_MODEL", "other/model"),
            ("INGEST_VERSION", 99),
        ):
            with self.subTest(attribute):
                record = self.store.list()[0] if self.store.list() else self.indexed()
                with mock.patch.object(rag_core, attribute, value):
                    restarted = self.reopened()
                    self.assertEqual(restarted.get(record.id).status, "failed")
                    self.assertIn("settings", restarted.get(record.id).error)

    def test_a_missing_index_is_reported_and_not_rebuilt(self):
        record = self.indexed()
        shutil.rmtree(self.root / record.id / "index")
        embeddings = CountingEmbeddings(size=16)
        restarted = self.reopened(embeddings)
        self.assertEqual(restarted.get(record.id).status, "failed")
        self.assertEqual(embeddings.calls, 0)

    def test_a_missing_pdf_is_reported(self):
        record = self.indexed()
        (self.root / record.id / "original.pdf").unlink()
        self.assertIn("missing", self.reopened().get(record.id).error)

    def test_a_damaged_index_file_fails_that_document_when_opened(self):
        damaged = self.indexed("alpha")
        healthy = self.indexed("beta")
        (self.root / damaged.id / "index" / "index.faiss").write_bytes(b"damaged")
        embeddings = CountingEmbeddings(size=16)
        restarted = self.reopened(embeddings)
        restarted.warm()
        self.assertEqual(restarted.get(damaged.id).status, "failed")
        self.assertEqual(restarted.get(healthy.id).status, "ready")
        self.assertEqual(embeddings.calls, 0)
        self.assertEqual((self.root / damaged.id / "index" / "index.faiss").read_bytes(), b"damaged")

    def test_an_unreadable_manifest_is_listed_as_failed_and_can_be_deleted(self):
        record = self.indexed()
        (self.root / record.id / "manifest.json").write_text("not json")
        restarted = self.reopened()
        self.assertEqual(restarted.get(record.id).status, "failed")
        restarted.delete(record.id)
        self.assertEqual(restarted.list(), [])

    def test_a_document_that_is_not_ready_cannot_be_opened(self):
        record = self.upload()
        with self.assertRaises(DocumentNotReady):
            self.store.open_index(record.id)

    def test_loading_ignores_unrelated_folders(self):
        self.root.mkdir(exist_ok=True)
        (self.root / ".incoming-abc").mkdir()
        (self.root / "not-an-id").mkdir()
        self.assertEqual(self.reopened().list(), [])


class CommandLineSupportTests(StoreTestCase):
    def test_adding_a_local_file_copies_its_suggested_questions(self):
        pdf = self.directory / "guide.pdf"
        write_pdf(pdf, topic_pages("alpha"))
        (self.directory / "guide.questions.json").write_text(json.dumps([{"question": "Why?", "note": "A note"}, {"question": "How?"}]))
        record = self.store.add_file(pdf)
        self.assertEqual(
            self.store.questions(record.id),
            [{"question": "Why?", "note": "A note"}, {"question": "How?", "note": None}],
        )
        self.assertEqual(len(self.reopened().questions(record.id)), 2)

    def test_invalid_suggested_questions_are_ignored(self):
        for content in ("not json", '{"question": "x"}', '[{"note": "no question"}]', "[1]"):
            pdf = self.directory / "guide.pdf"
            write_pdf(pdf, [content])
            (self.directory / "guide.questions.json").write_text(content)
            with self.assertLogs("document_store", "WARNING"):
                record = self.store.add_file(pdf)
            self.assertEqual(self.store.questions(record.id), [])

    def test_a_document_without_suggested_questions_has_none(self):
        self.assertEqual(self.store.questions(self.upload().id), [])

    def test_the_committed_sample_can_be_added_and_indexed(self):
        store = DocumentStore(self.root, embeddings=KeywordEmbeddings())
        store.load()
        record = store.index_now(store.add_file(rag_core.SAMPLE_DOCUMENT).id)
        self.assertEqual(record.status, "ready")
        self.assertEqual(record.pages_total, 4)
        self.assertTrue(store.questions(record.id))


class ConfigurationTests(unittest.TestCase):
    def test_the_documents_folder_is_configurable(self):
        with mock.patch.dict(os.environ, {"RAG_DOCUMENTS_DIR": "elsewhere"}):
            self.assertEqual(document_store.documents_root(), Path("elsewhere"))

    def test_the_upload_limit_is_configurable_and_has_a_default(self):
        with mock.patch.dict(os.environ, {"RAG_MAX_UPLOAD_MB": "2"}):
            self.assertEqual(document_store.max_upload_bytes(), 2 * 1024 * 1024)
        for value in ("", "not a number"):
            with mock.patch.dict(os.environ, {"RAG_MAX_UPLOAD_MB": value}):
                self.assertEqual(document_store.max_upload_bytes(), 50 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
