// Test-only data. Production components never import this file.
import type { DocumentSummary, QueryEvent, Source, Workspace } from "../api/types";

export function documentFixture(overrides: Partial<DocumentSummary> = {}): DocumentSummary {
  return {
    id: "doc-a",
    name: "sample.pdf",
    title: "Sample",
    status: "ready",
    size_bytes: 1000,
    pages_total: 534,
    pages_indexed: 504,
    pages_skipped: { blank: 30 },
    chunks: 2012,
    error: null,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    indexed_at: "2026-01-01T00:00:00+00:00",
    suggested_questions: [{ question: "What is an LSTM?", note: null }],
    ...overrides,
  };
}

export function workspaceFixture(
  index: Partial<Workspace["index"]> = {},
  documents: DocumentSummary[] = [documentFixture()],
): Workspace {
  return {
    index: { state: "ready", stage: null, error: null, elapsed_seconds: 12, ...index },
    documents,
    pipeline: {
      chunk_size_chars: 500,
      chunk_overlap_chars: 100,
      embedding_model: "org/embedder",
      vector_store: "FAISS, one per document",
      search_type: "MMR",
      k: 6,
      fetch_k: 20,
      lambda: 0.7,
      llm_model: "test-model",
    },
    capabilities: { upload: true, document_selection: true, conversation_memory: false },
    limits: { max_upload_bytes: 50 * 1024 * 1024 },
  };
}

export const SOURCES: Source[] = [
  { rank: 1, document_id: "doc-a", document: "sample.pdf", page: 523, distance: 0.38, text: "The forget gate decides what to discard." },
  { rank: 2, document_id: "doc-a", document: "sample.pdf", page: 230, distance: 0.61, text: "A sigmoid layer produces the bit vector." },
];

export function answeredEvents(answer = "It **discards** state.", sources: Source[] = SOURCES): QueryEvent[] {
  return [
    { type: "stage", stage: "retrieving" },
    { type: "sources", sources, retrieval_ms: 42 },
    { type: "stage", stage: "generating" },
    { type: "token", text: answer },
    { type: "done", outcome: "answered", generation_ms: 1200, model: "test-model" },
  ];
}

export function ndjsonResponse(events: QueryEvent[]): Response {
  return new Response(events.map((event) => JSON.stringify(event)).join("\n") + "\n", {
    headers: { "Content-Type": "application/x-ndjson" },
  });
}

export function stubApi(workspace: Workspace, query: () => Response | Promise<Response>) {
  const fetchMock = async (url: string) =>
    url === "/api/workspace" ? new Response(JSON.stringify(workspace)) : query();
  return fetchMock as unknown as typeof fetch;
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

export interface FakeApi {
  fetch: typeof fetch;
  calls: { url: string; method: string; body: unknown }[];
  setWorkspace: (workspace: Workspace) => void;
}

/** A small in-memory API. Handlers can replace any non-workspace route's response. */
export function fakeApi(
  initial: Workspace,
  handlers: Record<string, (body: unknown) => Response | Promise<Response>> = {},
): FakeApi {
  let workspace = initial;
  const calls: FakeApi["calls"] = [];
  const fetchMock = async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : init?.body;
    if (url === "/api/workspace") return new Response(JSON.stringify(workspace));
    calls.push({ url, method, body });
    const handler = handlers[`${method} ${url}`];
    return handler ? handler(body) : jsonResponse({ code: "bad_request", message: "Unhandled" }, 500);
  };
  return {
    fetch: fetchMock as unknown as typeof fetch,
    calls,
    setWorkspace: (next) => {
      workspace = next;
    },
  };
}
