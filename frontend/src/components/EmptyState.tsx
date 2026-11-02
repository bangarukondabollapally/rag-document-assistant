import type { SuggestedQuestion, Workspace } from "../api/types";
import type { DocumentsState } from "../state/useDocuments";
import { ErrorState } from "./ErrorState";
import { UploadButton } from "./UploadButton";
import styles from "./EmptyState.module.css";

interface EmptyStateProps {
  workspace: Workspace | null;
  unreachable: boolean;
  ready: boolean;
  documents: DocumentsState;
  onAsk: (question: string) => void;
}

export function EmptyState({ workspace, unreachable, ready, documents, onAsk }: EmptyStateProps) {
  if (unreachable) {
    return (
      <div className={styles.empty}>
        <ErrorState
          title="Can't reach the API"
          message="It can take a moment to start. If it isn't running, start it from the project root. This page reconnects automatically."
          command="uvicorn api:app --port 8000"
        />
      </div>
    );
  }

  if (workspace?.index.state === "failed") {
    return (
      <div className={styles.empty}>
        <ErrorState title="The index couldn't be opened" message={workspace.index.error ?? "Check the API logs."} />
      </div>
    );
  }

  // The first response normally arrives immediately; the input shows "Connecting" meanwhile.
  if (!workspace) return null;

  if (workspace.documents.length === 0) {
    return (
      <div className={styles.empty}>
        <h2 className={styles.heading}>No documents yet</h2>
        <p className={styles.flow}>Upload a PDF to get started</p>
        <div className={styles.emptyAction}>
          <UploadButton onUpload={(file) => void documents.upload(file)} />
        </div>
        {documents.error && (
          <p className={styles.uploadError} role="alert">
            {documents.error}
          </p>
        )}
      </div>
    );
  }

  const { selected } = documents;
  if (selected.length === 0) {
    const anyReady = workspace.documents.some((document) => document.status === "ready");
    return (
      <div className={styles.empty}>
        <h2 className={styles.heading}>{anyReady ? "Select a document" : "Your document isn't ready yet"}</h2>
        <p className={styles.flow}>
          {anyReady
            ? "Choose one or more documents to search. Only the documents you select are used."
            : "You can ask questions as soon as indexing finishes."}
        </p>
      </div>
    );
  }

  const seen = new Set<string>();
  const suggestions: SuggestedQuestion[] = selected
    .flatMap((document) => document.suggested_questions)
    .filter(({ question }) => !seen.has(question) && Boolean(seen.add(question)));

  return (
    <div className={styles.empty}>
      <h2 className={styles.heading}>
        Ask a question about {selected.length === 1 ? selected[0]?.name : `${selected.length} documents`}
      </h2>
      <ol className={styles.flow}>
        <li>Your question searches the indexed chunks of the selected documents, and no others.</li>
        <li>The {workspace.pipeline.k} selected passages are sent to the model as its context.</li>
        <li>The model answers from them, and the passages stay beside the answer for checking.</li>
      </ol>

      {workspace.index.state === "loading" && (
        <p className={styles.waiting} role="status">
          Loading the saved index. You can ask once it's ready.
        </p>
      )}

      {suggestions.length > 0 && (
        <div className={styles.suggestions}>
          <h3 className={styles.suggestionsLabel}>Try one</h3>
          <ul>
            {suggestions.map(({ question, note }) => (
              <li key={question}>
                <button type="button" className={styles.suggestion} disabled={!ready} onClick={() => onAsk(question)}>
                  <span>{question}</span>
                  {note && <span className={styles.suggestionNote}>{note}</span>}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
