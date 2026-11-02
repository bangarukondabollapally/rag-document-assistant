// Test-only data. Production components never import this file.
import type { QueryEvent, Source, Workspace } from "../api/types";

export function workspaceFixture(index: Partial<Workspace["index"]> = {}): Workspace {
  const merged: Workspace["index"] = {
    state: "ready",
    stage: null,
    reason: null,
    rebuild_command: null,
    pages: 504,
    chunks: 2012,
    error: null,
    elapsed_seconds: 12,
    ...index,
  };
  return {
    index: merged,
    documents: [
      {
        id: "sample.pdf",
        name: "sample.pdf",
        title: "Sample",
        status: merged.state,
        pages_total: 534,
        pages_indexed: 504,
        pages_skipped: { blank: 30 },
        chunks: 2012,
      },
    ],
    suggested_questions: [{ question: "What is an LSTM?", note: null }],
    pipeline: {
      chunk_size_chars: 500,
      chunk_overlap_chars: 100,
      embedding_model: "org/embedder",
      vector_store: "FAISS (in memory)",
      search_type: "MMR",
      k: 6,
      fetch_k: 20,
      lambda: 0.7,
      llm_model: "test-model",
    },
    capabilities: { upload: false, document_selection: false, conversation_memory: false },
  };
}

export const SOURCES: Source[] = [
  { rank: 1, document: "sample.pdf", page: 523, distance: 0.38, text: "The forget gate decides what to discard." },
  { rank: 2, document: "sample.pdf", page: 230, distance: 0.61, text: "A sigmoid layer produces the bit vector." },
];

export function answeredEvents(answer = "It **discards** state."): QueryEvent[] {
  return [
    { type: "stage", stage: "retrieving" },
    { type: "sources", sources: SOURCES, retrieval_ms: 42 },
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
