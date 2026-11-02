import type { ErrorCode, QueryError, QueryEvent, Workspace } from "./types";

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

export async function* streamQuery(question: string, signal: AbortSignal): AsyncGenerator<QueryEvent> {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!response.ok || !response.body) {
    if (response.status === 503) {
      throw new ApiError("not_ready", "The document index isn't ready.");
    }
    if (response.status === 422) {
      throw new ApiError("bad_request", "That question couldn't be sent. Questions can be up to 2,000 characters.");
    }
    throw new ApiError("network", `The API returned ${response.status}.`);
  }

  yield* readNdjson<QueryEvent>(response.body);
}
