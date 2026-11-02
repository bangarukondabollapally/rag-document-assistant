import type { QueryError } from "../api/types";
import styles from "./ErrorState.module.css";

interface ErrorStateProps {
  title: string;
  message: string;
  command?: string;
  onRetry?: () => void;
}

export function ErrorState({ title, message, command, onRetry }: ErrorStateProps) {
  return (
    <div className={styles.error} role="alert">
      <p className={styles.title}>{title}</p>
      <p className={styles.message}>{message}</p>
      {command && <code className={styles.command}>{command}</code>}
      {onRetry && (
        <button type="button" className={styles.retry} onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function describeQueryError(error: QueryError): { title: string; message: string } {
  switch (error.code) {
    case "rate_limited":
      return {
        title: "Rate limit reached",
        message: error.retry_after_seconds
          ? `${error.message} Try again in about ${error.retry_after_seconds} seconds.`
          : `${error.message} Wait a moment and try again.`,
      };
    case "auth":
      return { title: "API key rejected", message: `${error.message} Check GROQ_API_KEY in .env.` };
    case "model_unavailable":
      return { title: "Model unavailable", message: `${error.message} Set RAG_LLM_MODEL in .env and restart the API.` };
    case "retrieval_failed":
      return { title: "Search failed", message: error.message };
    case "network":
      return { title: "Connection problem", message: error.message };
    case "no_documents_selected":
      return { title: "No document selected", message: error.message };
    case "document_not_ready":
      return { title: "Document not ready", message: error.message };
    case "document_not_found":
      return { title: "Document not found", message: error.message };
    case "bad_request":
      return { title: "Question not sent", message: error.message };
    case "generation_failed":
      return { title: "No answer generated", message: error.message };
    default:
      return { title: "Request failed", message: error.message };
  }
}
