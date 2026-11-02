// Mirrors the responses of api.py. Later this can be generated from FastAPI's OpenAPI schema.

export type IndexState = "loading" | "ready" | "not_indexed" | "failed";

export interface IndexStatus {
  state: IndexState;
  stage: "checking" | "loading_index" | null;
  reason: string | null;
  rebuild_command: string | null;
  pages: number | null;
  chunks: number | null;
  error: string | null;
  elapsed_seconds: number;
}

export interface DocumentSummary {
  id: string;
  name: string;
  title: string | null;
  status: IndexState;
  pages_total: number | null;
  pages_indexed: number | null;
  pages_skipped: Record<string, number>;
  chunks: number | null;
}

export interface PipelineConfig {
  chunk_size_chars: number;
  chunk_overlap_chars: number;
  embedding_model: string;
  vector_store: string;
  search_type: string;
  k: number;
  fetch_k: number;
  lambda: number;
  llm_model: string;
}

export interface Capabilities {
  upload: boolean;
  document_selection: boolean;
  conversation_memory: boolean;
}

export interface SuggestedQuestion {
  question: string;
  note: string | null;
}

export interface Workspace {
  index: IndexStatus;
  documents: DocumentSummary[];
  suggested_questions: SuggestedQuestion[];
  pipeline: PipelineConfig;
  capabilities: Capabilities;
}

export interface Source {
  rank: number;
  document: string;
  page: number | null;
  distance: number;
  text: string;
}

export type ErrorCode =
  | "rate_limited"
  | "auth"
  | "model_unavailable"
  | "network"
  | "generation_failed"
  | "retrieval_failed"
  | "not_ready"
  | "bad_request";

export interface QueryError {
  code: ErrorCode;
  message: string;
  retry_after_seconds?: number | null;
}

export type QueryEvent =
  | { type: "stage"; stage: "retrieving" | "generating" }
  | { type: "sources"; sources: Source[]; retrieval_ms: number }
  | { type: "token"; text: string }
  | { type: "done"; outcome: "answered" | "not_found"; generation_ms: number; model: string }
  | ({ type: "error"; stage: "retrieving" | "generating" } & QueryError);
