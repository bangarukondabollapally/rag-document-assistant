"""API contract tests. They use a fake pipeline, so no model or network is needed."""

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

from langchain_core.embeddings import DeterministicFakeEmbedding

import api
import rag_core
from samples.make_sample import write_pdf


class CountingEmbeddings(DeterministicFakeEmbedding):
    calls: int = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def groq_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls("failure", response=response, body=None)


class FakePipeline:
    document = rag_core.DocumentInfo("doc.pdf", "Title", 10, 8, 3, {"blank": 2})

    def __init__(self, tokens=("Hello", " there"), error=None, retrieve_error=None):
        self.tokens, self.error, self.retrieve_error = tokens, error, retrieve_error

    def retrieve(self, question):
        if self.retrieve_error:
            raise self.retrieve_error
        return [rag_core.RetrievedChunk(1, "doc.pdf", "chunk text", 12, 0.5)]

    def stream(self, question, chunks):
        yield from self.tokens
        if self.error:
            raise self.error


class ApiTests(unittest.TestCase):
    def setUp(self):
        api.manager.pipeline = None
        self.client = TestClient(api.app)

    def post(self, pipeline):
        api.manager.pipeline = pipeline
        response = self.client.post("/api/query", json={"question": "Why?"})
        self.assertEqual(response.status_code, 200)
        return [json.loads(line) for line in response.text.splitlines()]

    def test_events_arrive_in_pipeline_order(self):
        events = self.post(FakePipeline())
        self.assertEqual(
            [e["type"] for e in events],
            ["stage", "sources", "stage", "token", "token", "done"],
        )
        self.assertEqual(events[1]["sources"][0]["page"], 12)
        self.assertEqual(events[-1]["outcome"], "answered")

    def test_refusal_is_reported_as_not_found(self):
        events = self.post(FakePipeline(tokens=("I couldn\u2019t find that", " in the document.")))
        self.assertEqual(events[-1]["outcome"], "not_found")

    def test_answer_mentioning_refusal_words_is_still_answered(self):
        events = self.post(FakePipeline(tokens=("I couldn't find that in the document, but X.",)))
        self.assertEqual(events[-1]["outcome"], "answered")

    def test_rate_limit_keeps_partial_answer_and_retry_time(self):
        error = groq_error(groq.RateLimitError, 429, {"retry-after": "5"})
        events = self.post(FakePipeline(error=error))
        self.assertEqual(events[-2]["type"], "token")
        self.assertEqual(events[-1]["code"], "rate_limited")
        self.assertEqual(events[-1]["retry_after_seconds"], 5)

    def test_missing_api_key_is_reported_on_the_first_question(self):
        events = self.post(FakePipeline(error=ValueError("The api_key client option must be set")))
        self.assertEqual(events[-1]["code"], "auth")
        self.assertIn("GROQ_API_KEY", events[-1]["message"])

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
        events = self.post(FakePipeline(error=groq_error(groq.NotFoundError, 404)))
        self.assertEqual(events[-1]["code"], "model_unavailable")

    def test_retrieval_failure_stops_before_generation(self):
        events = self.post(FakePipeline(retrieve_error=RuntimeError("boom")))
        self.assertEqual(events[-1]["code"], "retrieval_failed")
        self.assertNotIn("generating", [e.get("stage") for e in events[:-1]])

    def test_query_before_index_is_ready_returns_503(self):
        response = self.client.post("/api/query", json={"question": "Why?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "not_ready")

    def test_blank_question_is_rejected(self):
        api.manager.pipeline = FakePipeline()
        response = self.client.post("/api/query", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_workspace_reports_only_real_capabilities(self):
        capabilities = self.client.get("/api/workspace").json()["capabilities"]
        self.assertEqual(
            capabilities,
            {"upload": False, "document_selection": False, "conversation_memory": False},
        )


    def test_workspace_reports_the_ready_document(self):
        api.manager.pipeline = FakePipeline()
        document = self.client.get("/api/workspace").json()["documents"][0]
        self.assertEqual(document["pages_indexed"], 8)
        self.assertEqual(document["pages_total"], 10)
        self.assertEqual(document["chunks"], 3)
        self.assertEqual(document["pages_skipped"], {"blank": 2})

    def test_workspace_works_before_the_pipeline_is_ready(self):
        document = self.client.get("/api/workspace").json()["documents"][0]
        self.assertIsNone(document["title"])
        self.assertEqual(document["pages_skipped"], {})

    def test_workspace_reads_questions_for_the_current_document(self):
        questions = [{"question": "Why?", "note": None}]
        with mock.patch.object(rag_core, "suggested_questions", return_value=questions):
            self.assertEqual(self.client.get("/api/workspace").json()["suggested_questions"], questions)

    def test_workspace_exposes_unchanged_pipeline_settings(self):
        pipeline = self.client.get("/api/workspace").json()["pipeline"]
        self.assertEqual((pipeline["k"], pipeline["fetch_k"], pipeline["lambda"]), (6, 20, 0.7))
        self.assertEqual(pipeline["chunk_size_chars"], 500)


class IndexManagerTests(unittest.TestCase):
    def build(self, opener):
        manager = api.IndexManager(opener=opener)
        manager._run()
        return manager

    def test_success_records_ready(self):
        pipeline = FakePipeline()
        manager = self.build(lambda report: pipeline)
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["state"], "ready")
        self.assertIsNone(snapshot["stage"])
        self.assertIsNone(snapshot["rebuild_command"])
        self.assertIs(manager.pipeline, pipeline)

    def test_no_saved_index_reports_not_indexed_with_the_explicit_command(self):
        def opener(report):
            raise rag_core.IndexNotAvailable("The document has changed.")

        snapshot = self.build(opener).snapshot()
        self.assertEqual(snapshot["state"], "not_indexed")
        self.assertEqual(snapshot["reason"], "The document has changed.")
        self.assertEqual(snapshot["rebuild_command"], "python rag_core.py --rebuild")
        self.assertIsNone(snapshot["error"])

    def test_progress_starts_at_checking_and_follows_reported_stages(self):
        manager = api.IndexManager(opener=lambda report: FakePipeline())
        self.assertEqual(manager.snapshot()["state"], "loading")
        self.assertEqual(manager.snapshot()["stage"], "checking")
        manager._on_stage("loading_index", {"chunks": 3, "pages": 8})
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["stage"], "loading_index")
        self.assertEqual((snapshot["chunks"], snapshot["pages"]), (3, 8))

    def test_document_errors_are_shown_as_written(self):
        def builder(report):
            raise rag_core.DocumentError("doc.pdf has no extractable text.")

        self.assertEqual(self.build(builder).snapshot()["error"], "doc.pdf has no extractable text.")

    def test_failure_is_reported_without_leaking_internals(self):
        def builder(report):
            raise RuntimeError("secret internal path C:/x")

        snapshot = self.build(builder).snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertNotIn("secret", snapshot["error"])


class StartupLifecycleTests(unittest.TestCase):
    """Regression tests: starting the API must never process a document."""

    EXPENSIVE = ("load_document", "select_content_pages", "chunk_pages", "build_index", "rebuild_index")

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.document = self.directory / "doc.pdf"
        write_pdf(self.document, [f"Topic {i} covers learning method number {i}." for i in range(6)])
        self.embeddings = []
        environment = {"RAG_DOCUMENT_PATH": str(self.document), "RAG_INDEX_DIR": str(self.directory / "index")}
        for patcher in (
            mock.patch.dict(os.environ, environment),
            mock.patch.object(api, "manager", api.IndexManager()),
            mock.patch.object(rag_core, "_make_embeddings", side_effect=self.make_embeddings),
            # Fails the test if startup tries to create the LLM.
            mock.patch.object(rag_core, "_make_llm", side_effect=AssertionError("startup created the LLM")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_embeddings(self):
        embeddings = CountingEmbeddings(size=16)
        self.embeddings.append(embeddings)
        return embeddings

    def forbid_ingestion(self):
        patchers = [mock.patch.object(rag_core, name, side_effect=AssertionError(name)) for name in self.EXPENSIVE]
        patchers.append(mock.patch.object(rag_core.FAISS, "from_documents", side_effect=AssertionError("from_documents")))
        mocks = [patcher.start() for patcher in patchers]
        for patcher in patchers:
            self.addCleanup(patcher.stop)
        return mocks

    def start_api(self):
        """Runs the real lifespan and waits for its background thread to finish."""
        client = TestClient(api.app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        deadline = time.time() + 20
        while api.manager.snapshot()["state"] == "loading" and time.time() < deadline:
            time.sleep(0.01)
        return client

    def assert_nothing_expensive_ran(self, mocks):
        for called in mocks:
            called.assert_not_called()
        self.assertEqual(sum(e.calls for e in self.embeddings), 0)

    def test_a_existing_index_is_loaded_without_any_ingestion(self):
        rag_core.rebuild_index(self.document, embeddings=CountingEmbeddings(size=16))
        mocks = self.forbid_ingestion()

        workspace = self.start_api().get("/api/workspace").json()

        self.assertEqual(workspace["index"]["state"], "ready")
        self.assertIsNotNone(api.manager.pipeline)
        self.assertEqual(workspace["documents"][0]["pages_total"], 6)
        self.assert_nothing_expensive_ran(mocks)

    def test_a_loaded_index_answers_retrieval_without_ingestion(self):
        rag_core.rebuild_index(self.document, embeddings=CountingEmbeddings(size=16))
        mocks = self.forbid_ingestion()
        self.start_api()
        self.assertEqual(len(api.manager.pipeline.retrieve("Topic 3")), 6)
        self.assert_nothing_expensive_ran(mocks)

    def test_b_no_index_reports_not_indexed_without_any_ingestion(self):
        mocks = self.forbid_ingestion()
        client = self.start_api()

        workspace = client.get("/api/workspace").json()

        self.assertEqual(workspace["index"]["state"], "not_indexed")
        self.assertIn("No saved index", workspace["index"]["reason"])
        self.assertEqual(workspace["documents"][0]["status"], "not_indexed")
        self.assertEqual(client.post("/api/query", json={"question": "Why?"}).status_code, 503)
        self.assertFalse((self.directory / "index").exists())
        self.assert_nothing_expensive_ran(mocks)

    def test_b_a_damaged_index_is_not_rebuilt_automatically(self):
        rag_core.rebuild_index(self.document, embeddings=CountingEmbeddings(size=16))
        check = rag_core.check_index(self.document)
        (check.directory / "index.faiss").write_bytes(b"damaged")
        mocks = self.forbid_ingestion()

        self.start_api()

        self.assertEqual(api.manager.snapshot()["state"], "not_indexed")
        self.assertEqual((check.directory / "index.faiss").read_bytes(), b"damaged")
        self.assert_nothing_expensive_ran(mocks)

    def test_c_only_an_explicit_rebuild_runs_every_ingestion_step(self):
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
        embeddings = CountingEmbeddings(size=16)

        self.assertEqual(rag_core.check_index(self.document).status, "missing")
        rag_core.rebuild_index(self.document, embeddings=embeddings)

        self.assertEqual(steps, ["load_document", "select_content_pages", "chunk_pages", "from_documents", "save_local"])
        self.assertEqual(embeddings.calls, 1)
        self.assertEqual(rag_core.check_index(self.document).status, "valid")

    def test_reading_workspace_status_never_triggers_work(self):
        mocks = self.forbid_ingestion()
        client = self.start_api()
        for _ in range(3):
            client.get("/api/workspace")
        self.assert_nothing_expensive_ran(mocks)

    def test_importing_the_core_does_not_load_the_heavy_ml_libraries(self):
        code = (
            "import sys, rag_core; "
            "heavy = [m for m in ('torch', 'transformers', 'sentence_transformers', 'langchain_text_splitters', "
            "'langchain_groq') if m in sys.modules]; "
            "sys.exit(repr(heavy) if heavy else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=rag_core.ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-300:])


if __name__ == "__main__":
    unittest.main()
