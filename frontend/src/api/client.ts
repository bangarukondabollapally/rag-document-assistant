import type { DocumentSummary, ErrorCode, QueryError, QueryEvent, Workspace } from "./types";

export class ApiError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string) {
    super(message);
    this.code = code;
  }

  toQueryError(): QueryError {
    return { code: this.code, message: this.message };
  }
}

export async function fetchWorkspace(signal?: AbortSignal): Promise<Workspace> {
  const response = await fetch("/api/workspace", { signal });
  if (!response.ok) {
    throw new ApiError("network", `The API returned ${response.status}.`);
  }
  return (await response.json()) as Workspace;
}

/** Yields one parsed object per line, buffering lines split across network chunks. */
export async function* readNdjson<T>(stream: ReadableStream<Uint8Array>): AsyncGenerator<T> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.trim()) yield JSON.parse(line) as T;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield JSON.parse(buffer) as T;
  } finally {
    reader.releaseLock();
  }
}

/** Turns the API's {code, message} error body into an ApiError, whatever its shape. */
async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as { code?: ErrorCode; message?: string };
    if (body.code && body.message) return new ApiError(body.code, body.message);
  } catch {
    // Not a JSON error body: fall through to the generic message.
  }
  return new ApiError("network", `The API returned ${response.status}.`);
}

export async function* streamQuery(
  question: string,
  selectedDocumentIds: string[],
  signal: AbortSignal,
): AsyncGenerator<QueryEvent> {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, selected_document_ids: selectedDocumentIds }),
    signal,
  });

  if (response.status === 422) {
    throw new ApiError("bad_request", "That question couldn't be sent. Questions can be up to 2,000 characters.");
  }
  if (!response.ok || !response.body) throw await toApiError(response);

  yield* readNdjson<QueryEvent>(response.body);
}

export async function uploadDocument(file: File): Promise<DocumentSummary> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/documents", { method: "POST", body: form });
  if (!response.ok) throw await toApiError(response);
  return (await response.json()) as DocumentSummary;
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`/api/documents/${id}`, { method: "DELETE" });
  if (!response.ok) throw await toApiError(response);
}

export async function reindexDocument(id: string): Promise<void> {
  const response = await fetch(`/api/documents/${id}/reindex`, { method: "POST" });
  if (!response.ok) throw await toApiError(response);
}
