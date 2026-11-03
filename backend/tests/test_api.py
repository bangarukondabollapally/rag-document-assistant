"""API contract tests, using real stores with fake embeddings and a fake LLM."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import groq
import httpx
from fastapi.testclient import TestClient

import api
import document_store
import rag_core
from tests.support import CountingEmbeddings, FakeLLM, KeywordEmbeddings, pdf_bytes, topic_pages


def groq_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls("failure", response=response, body=None)


class FakePipeline:
    """Lets tests control what generation does, without a model."""

    def __init__(self, tokens=("Hello", " there"), error=None, retrieve_error=None):
        self.tokens, self.error, self.retrieve_error = tokens, error, retrieve_error
        self.searched = None

    def retrieve(self, documents, question):
        self.searched = [document.id for document in documents]
        if self.retrieve_error:
            raise self.retrieve_error
        return [rag_core.RetrievedChunk(1, documents[0].id, documents[0].name, "chunk text", 12, 0.5)]

    def stream(self, question, chunks):
        yield from self.tokens
        if self.error:
            raise self.error


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.embeddings = KeywordEmbeddings()
        self.store = document_store.DocumentStore(self.directory / "documents", embeddings=self.embeddings)
        self.store.load()
        for name, value in (("store", self.store), ("engine", api.SearchEngine(self.store))):
            patcher = mock.patch.object(api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(api.app)

    def use_pipeline(self, pipeline):
        patcher = mock.patch.object(api, "pipeline", pipeline)
        patcher.start()
        self.addCleanup(patcher.stop)
        return pipeline

    def upload(self, keyword="alpha", filename=None, pages=None):
        body = pdf_bytes(pages or topic_pages(keyword))
        response = self.client.post(
            "/api/documents",
            files={"file": (filename or f"{keyword}.pdf", body, "application/pdf")},
        )
        self.store.wait_until_idle()
        return response

    def ready_document(self, keyword="alpha", filename=None, pages=None):
        response = self.upload(keyword, filename, pages)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def ask(self, ids, question="Why?"):
        response = self.client.post("/api/query", json={"question": question, "selected_document_ids": ids})
        self.assertEqual(response.status_code, 200, response.text)
        return [json.loads(line) for line in response.text.splitlines()]


class QueryStreamTests(ApiTestCase):
    def ask_with(self, pipeline):
        self.use_pipeline(pipeline)
        return self.ask([self.ready_document()])

    def test_events_arrive_in_pipeline_order(self):
        events = self.ask_with(FakePipeline())
        self.assertEqual(
            [e["type"] for e in events],
            ["stage", "sources", "stage", "token", "token", "done"],
        )
        self.assertEqual(events[1]["sources"][0]["page"], 12)
        self.assertEqual(events[-1]["outcome"], "answered")

    def test_refusal_is_reported_as_not_found(self):
        events = self.ask_with(FakePipeline(tokens=("I couldn\u2019t find that", " in the document.")))
        self.assertEqual(events[-1]["outcome"], "not_found")

    def test_answer_mentioning_refusal_words_is_still_answered(self):
        events = self.ask_with(FakePipeline(tokens=("I couldn't find that in the document, but X.",)))
        self.assertEqual(events[-1]["outcome"], "answered")

    def test_rate_limit_keeps_partial_answer_and_retry_time(self):
        error = groq_error(groq.RateLimitError, 429, {"retry-after": "5"})
        events = self.ask_with(FakePipeline(error=error))
        self.assertEqual(events[-2]["type"], "token")
        self.assertEqual(events[-1]["code"], "rate_limited")
        self.assertEqual(events[-1]["retry_after_seconds"], 5)

    def test_missing_api_key_is_reported_on_the_first_question(self):
        events = self.ask_with(FakePipeline(error=ValueError("The api_key client option must be set")))
        self.assertEqual(events[-1]["code"], "auth")
        self.assertIn("GROQ_API_KEY", events[-1]["message"])

    def test_a_rejected_api_key_is_not_described_as_missing(self):
        request = httpx.Request("POST", "https://api.groq.com")
        response = httpx.Response(401, request=request)
        error = groq.AuthenticationError("Error code: 401 - invalid_api_key", response=response, body=None)
        self.assertEqual(api._llm_error(error), {"code": "auth", "message": "The Groq API key was rejected."})

    def test_blank_or_absent_api_key_is_reported_as_a_key_problem(self):
        for environment in ({"GROQ_API_KEY": ""}, {"GROQ_API_KEY": "   "}):
            with self.subTest(environment), mock.patch.dict(os.environ, environment):
                with self.assertRaises(ValueError) as caught:
                    rag_core._make_llm()
                self.assertEqual(api._llm_error(caught.exception)["code"], "auth")
        with mock.patch.dict(os.environ):
            os.environ.pop("GROQ_API_KEY", None)
            with self.assertRaises(ValueError) as caught:
                rag_core._make_llm()
            self.assertEqual(api._llm_error(caught.exception)["code"], "auth")

    def test_retired_model_has_a_specific_error(self):
        events = self.ask_with(FakePipeline(error=groq_error(groq.NotFoundError, 404)))
        self.assertEqual(events[-1]["code"], "model_unavailable")

    def test_retrieval_failure_stops_before_generation(self):
        events = self.ask_with(FakePipeline(retrieve_error=RuntimeError("boom")))
        self.assertEqual(events[-1]["code"], "retrieval_failed")
        self.assertNotIn("generating", [e.get("stage") for e in events[:-1]])

    def test_blank_question_is_rejected(self):
        document_id = self.ready_document()
        response = self.client.post("/api/query", json={"question": "   ", "selected_document_ids": [document_id]})
        self.assertEqual(response.status_code, 422)


class SelectedDocumentRetrievalTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.use_pipeline(rag_core.RagPipeline(llm=FakeLLM()))
        self.alpha = self.ready_document("alpha")
        self.beta = self.ready_document("beta")

    def sources(self, ids, question):
        return next(e["sources"] for e in self.ask(ids, question) if e["type"] == "sources")

    def test_one_selected_document_is_the_only_one_searched(self):
        sources = self.sources([self.beta], "alpha")  # The question matches the other document.
        self.assertTrue(sources)
        self.assertEqual({source["document_id"] for source in sources}, {self.beta})

    def test_multiple_selected_documents_are_searched_together(self):
        sources = self.sources([self.alpha, self.beta], "alpha beta")
        self.assertEqual({source["document_id"] for source in sources}, {self.alpha, self.beta})

    def test_an_unselected_document_is_never_included(self):
        gamma = self.ready_document("gamma")
        sources = self.sources([self.alpha, self.beta], "gamma")
        self.assertNotIn(gamma, {source["document_id"] for source in sources})

    def test_no_selected_documents_returns_a_clear_error_and_searches_nothing(self):
        pipeline = self.use_pipeline(FakePipeline())
        for body in ({"question": "Why?"}, {"question": "Why?", "selected_document_ids": []}):
            response = self.client.post("/api/query", json=body)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["code"], "no_documents_selected")
        self.assertIsNone(pipeline.searched)

    def test_an_unknown_document_is_refused(self):
        response = self.client.post("/api/query", json={"question": "Why?", "selected_document_ids": ["0" * 32]})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "document_not_found")

    def test_a_document_that_is_not_ready_is_refused_even_alongside_a_ready_one(self):
        pipeline = self.use_pipeline(FakePipeline())
        record = self.store.add_upload(__import__("io").BytesIO(pdf_bytes(topic_pages("delta"))), "delta.pdf", "application/pdf")
        response = self.client.post(
            "/api/query", json={"question": "Why?", "selected_document_ids": [self.alpha, record.id]}
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "document_not_ready")
        self.assertIn("delta.pdf", response.json()["message"])
        self.assertIsNone(pipeline.searched)

    def test_duplicate_selected_ids_search_the_document_once(self):
        pipeline = self.use_pipeline(FakePipeline())
        self.ask([self.alpha, self.alpha])
        self.assertEqual(pipeline.searched, [self.alpha])

    def test_sources_carry_document_page_score_and_text_from_retrieval(self):
        for source in self.sources([self.alpha, self.beta], "alpha beta"):
            self.assertEqual(set(source), {"rank", "document_id", "document", "text", "page", "distance"})
            expected_name = "alpha.pdf" if source["document_id"] == self.alpha else "beta.pdf"
            self.assertEqual(source["document"], expected_name)
            self.assertIn(source["page"], range(1, 5))
            self.assertIn(f"part {source['page']}", source["text"])
            self.assertIsInstance(source["distance"], float)

    def test_sources_arrive_before_the_answer_tokens(self):
        events = self.ask([self.alpha], "alpha")
        types = [event["type"] for event in events]
        self.assertLess(types.index("sources"), types.index("token"))

    def test_a_document_damaged_since_startup_gives_a_useful_error(self):
        restarted = document_store.DocumentStore(self.store.root, embeddings=KeywordEmbeddings())
        restarted.load()
        (restarted.root / self.alpha / "index" / "index.faiss").write_bytes(b"damaged")
        with mock.patch.object(api, "store", restarted):
            events = self.ask([self.alpha], "alpha")
        self.assertEqual(events[-1]["code"], "document_not_ready")
        self.assertEqual(restarted.get(self.alpha).status, "failed")


class UploadApiTests(ApiTestCase):
    def post_file(self, filename, body, content_type="application/pdf"):
        return self.client.post("/api/documents", files={"file": (filename, body, content_type)})

    def test_a_valid_pdf_is_stored_then_indexed_in_the_background(self):
        response = self.upload("alpha", "My Paper.pdf")
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["name"], "My Paper.pdf")
        self.assertRegex(body["id"], r"^[0-9a-f]{32}$")
        document = self.client.get(f"/api/documents/{body['id']}").json()
        self.assertEqual(document["status"], "ready")
        self.assertGreater(document["chunks"], 0)

    def test_upload_returns_before_indexing_finishes(self):
        started = []

        def slow_build(*args, **kwargs):
            started.append(True)
            raise rag_core.DocumentError("stopped by the test")

        release = mock.patch.object(rag_core, "build_index", side_effect=slow_build)
        release.start()
        self.addCleanup(release.stop)
        response = self.post_file("a.pdf", pdf_bytes(["x"]))
        self.assertEqual(response.status_code, 201)
        self.assertIn(response.json()["status"], ("uploaded", "indexing", "failed"))
        self.store.wait_until_idle()
        self.assertEqual(self.client.get(f"/api/documents/{response.json()['id']}").json()["status"], "failed")

    def test_an_invalid_file_type_is_rejected(self):
        response = self.post_file("notes.txt", b"plain text", "text/plain")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["code"], "invalid_type")
        self.assertEqual(self.client.get("/api/documents").json()["documents"], [])

    def test_a_fake_pdf_is_rejected(self):
        self.assertEqual(self.post_file("fake.pdf", b"<script>alert(1)</script>").status_code, 415)

    def test_an_oversized_file_is_rejected(self):
        with mock.patch.dict(os.environ, {"RAG_MAX_UPLOAD_MB": "0.001"}):
            response = self.post_file("big.pdf", b"%PDF-1.4\n" + b"x" * 5000)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["code"], "too_large")

    def test_a_very_large_declared_size_is_rejected_before_reading_the_body(self):
        with mock.patch.dict(os.environ, {"RAG_MAX_UPLOAD_MB": "1"}):
            response = self.client.post(
                "/api/documents",
                content=b"x",
                headers={"Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(50 * 1024 * 1024)},
            )
        self.assertEqual(response.status_code, 413)

    def test_a_duplicate_is_rejected_with_the_existing_document(self):
        first = self.ready_document("alpha", "first.pdf")
        response = self.upload("alpha", "second.pdf")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "duplicate")
        self.assertEqual(response.json()["document_id"], first)
        self.assertEqual(len(self.client.get("/api/documents").json()["documents"]), 1)

    def test_an_unsafe_filename_stays_inside_the_store(self):
        response = self.upload("alpha", "../../../../evil.pdf")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["name"], "evil.pdf")
        self.assertFalse((self.directory / "evil.pdf").exists())
        self.assertTrue((self.store.root / response.json()["id"] / "original.pdf").is_file())

    def test_a_request_without_a_file_is_rejected(self):
        response = self.client.post("/api/documents", data={"other": "value"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_upload")


class DocumentApiTests(ApiTestCase):
    def test_workspace_lists_real_documents_and_capabilities(self):
        self.assertEqual(self.client.get("/api/workspace").json()["documents"], [])
        document_id = self.ready_document("alpha")
        workspace = self.client.get("/api/workspace").json()
        self.assertEqual([document["id"] for document in workspace["documents"]], [document_id])
        self.assertEqual(
            workspace["capabilities"],
            {"upload": True, "document_selection": True, "conversation_memory": False},
        )
        self.assertEqual(workspace["limits"]["max_upload_bytes"], document_store.max_upload_bytes())

    def test_workspace_exposes_unchanged_pipeline_settings(self):
        pipeline = self.client.get("/api/workspace").json()["pipeline"]
        self.assertEqual((pipeline["k"], pipeline["fetch_k"], pipeline["lambda"]), (6, 20, 0.7))
        self.assertEqual((pipeline["chunk_size_chars"], pipeline["chunk_overlap_chars"]), (500, 100))

    def test_public_metadata_does_not_expose_local_paths(self):
        self.ready_document("alpha")
        text = self.client.get("/api/workspace").text
        self.assertNotIn(str(self.directory).replace("\\", "\\\\"), text)
        self.assertNotIn("original.pdf", text)

    def test_getting_an_unknown_or_unsafe_document_is_not_found(self):
        for document_id in ("0" * 32, "..%2F..%2Fsecret", "not-an-id"):
            response = self.client.get(f"/api/documents/{document_id}")
            self.assertEqual(response.status_code, 404, document_id)

    def test_deleting_removes_the_document_and_its_files(self):
        document_id = self.ready_document("alpha")
        self.assertEqual(self.client.delete(f"/api/documents/{document_id}").status_code, 204)
        self.assertEqual(self.client.get("/api/documents").json()["documents"], [])
        self.assertFalse((self.store.root / document_id).exists())
        response = self.client.post("/api/query", json={"question": "Why?", "selected_document_ids": [document_id]})
        self.assertEqual(response.status_code, 404)

    def test_deleting_an_unknown_document_is_not_found(self):
        self.assertEqual(self.client.delete(f"/api/documents/{'0' * 32}").status_code, 404)

    def test_deleting_during_indexing_is_refused(self):
        document_id = self.ready_document("alpha")
        self.store._busy.add(document_id)
        self.addCleanup(self.store._busy.discard, document_id)
        self.assertEqual(self.client.delete(f"/api/documents/{document_id}").status_code, 409)
        self.assertTrue((self.store.root / document_id).exists())

    def test_reindexing_rebuilds_only_the_requested_document(self):
        target = self.ready_document("alpha")
        other = self.ready_document("beta")
        other_manifest = (self.store.root / other / "manifest.json").read_text()
        before = self.embeddings.document_calls

        response = self.client.post(f"/api/documents/{target}/reindex")
        self.store.wait_until_idle()

        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.embeddings.document_calls, before + 1)
        self.assertEqual(self.client.get(f"/api/documents/{target}").json()["status"], "ready")
        self.assertEqual((self.store.root / other / "manifest.json").read_text(), other_manifest)

    def test_reindexing_an_unknown_document_is_not_found(self):
        self.assertEqual(self.client.post(f"/api/documents/{'0' * 32}/reindex").status_code, 404)

    def test_reindexing_during_indexing_is_refused(self):
        document_id = self.ready_document("alpha")
        self.store._busy.add(document_id)
        self.addCleanup(self.store._busy.discard, document_id)
        self.assertEqual(self.client.post(f"/api/documents/{document_id}/reindex").status_code, 409)

    def test_a_failed_document_reports_why_and_can_be_retried(self):
        with mock.patch.object(rag_core.FAISS, "from_documents", side_effect=RuntimeError("temporary")):
            document_id = self.upload("alpha").json()["id"]
        failed = self.client.get(f"/api/documents/{document_id}").json()
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["error"])
        self.client.post(f"/api/documents/{document_id}/reindex")
        self.store.wait_until_idle()
        self.assertEqual(self.client.get(f"/api/documents/{document_id}").json()["status"], "ready")


class SearchEngineTests(unittest.TestCase):
    def test_starts_loading_then_reports_ready(self):
        documents = mock.Mock()
        engine = api.SearchEngine(documents)
        self.assertEqual(engine.snapshot()["state"], "loading")
        engine._run()
        self.assertEqual(engine.snapshot()["state"], "ready")
        self.assertIsNone(engine.snapshot()["stage"])

    def test_failure_is_reported_without_leaking_internals(self):
        documents = mock.Mock()
        documents.warm.side_effect = RuntimeError("secret internal path C:/x")
        engine = api.SearchEngine(documents)
        with self.assertLogs("api", "ERROR"):
            engine._run()
        snapshot = engine.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertNotIn("secret", snapshot["error"])


class StartupLifecycleTests(unittest.TestCase):
    """Regression tests: starting the API must never process a document."""

    EXPENSIVE = ("load_document", "select_content_pages", "chunk_pages", "build_index", "_make_llm")

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.root = self.directory / "documents"

    def indexed_documents(self, *keywords):
        store = document_store.DocumentStore(self.root, embeddings=KeywordEmbeddings())
        store.load()
        ids = []
        for keyword in keywords:
            path = self.directory / f"{keyword}.pdf"
            path.write_bytes(pdf_bytes(topic_pages(keyword)))
            ids.append(store.index_now(store.add_file(path).id).id)
        return ids

    def forbid_ingestion(self):
        patchers = [mock.patch.object(rag_core, name, side_effect=AssertionError(name)) for name in self.EXPENSIVE]
        patchers.append(mock.patch.object(rag_core.FAISS, "from_documents", side_effect=AssertionError("from_documents")))
        mocks = [patcher.start() for patcher in patchers]
        for patcher in patchers:
            self.addCleanup(patcher.stop)
        return mocks

    def start_api(self, embeddings):
        """Runs the real lifespan and waits for its background thread to finish."""
        store = document_store.DocumentStore(self.root, embeddings=embeddings)
        for name, value in (("store", store), ("engine", api.SearchEngine(store))):
            patcher = mock.patch.object(api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        client = TestClient(api.app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        deadline = time.time() + 20
        while api.engine.snapshot()["state"] == "loading" and time.time() < deadline:
            time.sleep(0.01)
        return client, store

    def assert_nothing_expensive_ran(self, mocks, embeddings):
        for called in mocks:
            called.assert_not_called()
        self.assertEqual(embeddings.calls, 0)

    def test_existing_indexes_load_with_no_parsing_chunking_embedding_or_rebuild(self):
        ids = self.indexed_documents("alpha", "beta")
        before = {p: p.stat().st_mtime_ns for p in self.root.rglob("*") if p.is_file()}
        mocks = self.forbid_ingestion()
        embeddings = CountingEmbeddings(size=5)

        client, store = self.start_api(embeddings)
        workspace = client.get("/api/workspace").json()

        self.assertEqual(workspace["index"]["state"], "ready")
        self.assertEqual({document["status"] for document in workspace["documents"]}, {"ready"})
        self.assertEqual(set(store._indexes), set(ids))
        self.assert_nothing_expensive_ran(mocks, embeddings)
        self.assertEqual({p: p.stat().st_mtime_ns for p in self.root.rglob("*") if p.is_file()}, before)

    def test_an_opened_index_answers_retrieval_without_ingestion(self):
        ids = self.indexed_documents("alpha")
        mocks = self.forbid_ingestion()
        embeddings = CountingEmbeddings(size=5)
        _, store = self.start_api(embeddings)
        self.assertGreater(len(rag_core.retrieve(store.searchable(ids), "alpha")), 0)
        self.assert_nothing_expensive_ran(mocks, embeddings)

    def test_with_no_documents_startup_loads_no_model_at_all(self):
        mocks = self.forbid_ingestion()
        with mock.patch.object(rag_core, "_make_embeddings", side_effect=AssertionError("loaded the model")):
            store = document_store.DocumentStore(self.root)
            for name, value in (("store", store), ("engine", api.SearchEngine(store))):
                patcher = mock.patch.object(api, name, value)
                patcher.start()
                self.addCleanup(patcher.stop)
            with TestClient(api.app) as client:
                deadline = time.time() + 5
                while api.engine.snapshot()["state"] == "loading" and time.time() < deadline:
                    time.sleep(0.01)
                workspace = client.get("/api/workspace").json()
        self.assertEqual(workspace["index"]["state"], "ready")
        self.assertEqual(workspace["documents"], [])
        for called in mocks:
            called.assert_not_called()

    def test_a_missing_index_is_reported_and_nothing_rebuilds_it(self):
        ids = self.indexed_documents("alpha", "beta")
        shutil.rmtree(self.root / ids[0] / "index")
        mocks = self.forbid_ingestion()
        embeddings = CountingEmbeddings(size=5)

        client, _ = self.start_api(embeddings)
        statuses = {d["id"]: d["status"] for d in client.get("/api/workspace").json()["documents"]}

        self.assertEqual(statuses, {ids[0]: "failed", ids[1]: "ready"})
        self.assertFalse((self.root / ids[0] / "index").exists())
        self.assert_nothing_expensive_ran(mocks, embeddings)

    def test_a_damaged_index_is_reported_and_nothing_rebuilds_it(self):
        ids = self.indexed_documents("alpha")
        (self.root / ids[0] / "index" / "index.faiss").write_bytes(b"damaged")
        mocks = self.forbid_ingestion()
        embeddings = CountingEmbeddings(size=5)

        client, _ = self.start_api(embeddings)
        document = client.get(f"/api/documents/{ids[0]}").json()

        self.assertEqual(document["status"], "failed")
        self.assertIn("Reindex", document["error"])
        self.assertEqual((self.root / ids[0] / "index" / "index.faiss").read_bytes(), b"damaged")
        self.assert_nothing_expensive_ran(mocks, embeddings)

    def test_interrupted_indexing_stays_failed_until_explicitly_retried(self):
        ids = self.indexed_documents("alpha")
        manifest = self.root / ids[0] / "manifest.json"
        manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "status": "indexing"}))
        mocks = self.forbid_ingestion()
        embeddings = CountingEmbeddings(size=5)

        client, _ = self.start_api(embeddings)

        self.assertEqual(client.get(f"/api/documents/{ids[0]}").json()["status"], "failed")
        self.assert_nothing_expensive_ran(mocks, embeddings)

    def test_only_an_explicit_operation_runs_every_ingestion_step(self):
        steps = []

        def spy(owner, name):
            original = getattr(owner, name)

            def wrapper(*args, **kwargs):
                steps.append(name)
                return original(*args, **kwargs)

            # A plain function, so save_local still receives the instance as self.
            patcher = mock.patch.object(owner, name, new=wrapper)
            patcher.start()
            self.addCleanup(patcher.stop)

        for name in ("load_document", "select_content_pages", "chunk_pages"):
            spy(rag_core, name)
        spy(rag_core.FAISS, "from_documents")
        spy(rag_core.FAISS, "save_local")

        self.indexed_documents("alpha")

        self.assertEqual(steps, ["load_document", "select_content_pages", "chunk_pages", "from_documents", "save_local"])

    def test_reading_workspace_status_never_triggers_work(self):
        self.indexed_documents("alpha")
        mocks = self.forbid_ingestion()
        embeddings = CountingEmbeddings(size=5)
        client, _ = self.start_api(embeddings)
        for _ in range(3):
            client.get("/api/workspace")
            client.get("/api/documents")
        self.assert_nothing_expensive_ran(mocks, embeddings)

    def test_importing_the_api_does_not_load_the_heavy_ml_libraries(self):
        code = (
            "import sys, api; "
            "heavy = [m for m in ('torch', 'transformers', 'sentence_transformers', 'langchain_text_splitters', "
            "'langchain_groq') if m in sys.modules]; "
            "sys.exit(repr(heavy) if heavy else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=rag_core.ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-300:])


if __name__ == "__main__":
    unittest.main()
