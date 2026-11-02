// Mirrors the responses of api.py. Later this can be generated from FastAPI's OpenAPI schema.

export type EngineState = "loading" | "ready" | "failed";

export interface IndexStatus {
  state: EngineState;
  stage: "starting" | "loading_model" | "opening_indexes" | null;
  error: string | null;
  elapsed_seconds: number;
}

export type DocumentStatus = "uploaded" | "indexing" | "ready" | "failed";

export interface SuggestedQuestion {
  question: string;
  note: string | null;
}

export interface DocumentSummary {
  id: string;
  name: string;
  title: string | null;
  status: DocumentStatus;
  size_bytes: number;
  pages_total: number | null;
  pages_indexed: number | null;
  pages_skipped: Record<string, number>;
  chunks: number | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  indexed_at: string | null;
  suggested_questions: SuggestedQuestion[];
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

export interface Workspace {
  index: IndexStatus;
  documents: DocumentSummary[];
  pipeline: PipelineConfig;
  capabilities: Capabilities;
  limits: { max_upload_bytes: number };
}

export interface Source {
  rank: number;
  document_id: string;
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
  | "no_documents_selected"
  | "document_not_ready"
  | "document_not_found"
  | "document_busy"
  | "too_large"
  | "duplicate"
  | "invalid_type"
  | "invalid_upload"
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
