import type { Workspace } from "../api/types";
import { ErrorState } from "./ErrorState";
import styles from "./EmptyState.module.css";

interface EmptyStateProps {
  workspace: Workspace | null;
  unreachable: boolean;
  ready: boolean;
  onAsk: (question: string) => void;
}

export function EmptyState({ workspace, unreachable, ready, onAsk }: EmptyStateProps) {
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

  if (workspace?.index.state === "not_indexed") {
    return (
      <div className={styles.empty}>
        <ErrorState
          title="No index available"
          message={`${workspace.index.reason ?? ""} Build the index from the project root, then restart the API.`.trim()}
          command={workspace.index.rebuild_command ?? undefined}
        />
      </div>
    );
  }

  // The first response normally arrives immediately; the input shows "Connecting" meanwhile.
  if (!workspace) return null;

  return (
    <div className={styles.empty}>
      <h2 className={styles.heading}>Ask a question about {workspace.documents[0]?.name}</h2>
      <ol className={styles.flow}>
        <li>Your question searches the document's indexed chunks.</li>
        <li>The {workspace.pipeline.k} selected passages are sent to the model as its context.</li>
        <li>The model answers from them, and the passages stay beside the answer for checking.</li>
      </ol>

      {workspace.index.state === "loading" && (
        <p className={styles.waiting} role="status">
          Loading the saved index. You can ask once it's ready.
        </p>
      )}

      {workspace.suggested_questions.length > 0 && (
        <div className={styles.suggestions}>
          <h3 className={styles.suggestionsLabel}>Try one</h3>
          <ul>
            {workspace.suggested_questions.map(({ question, note }) => (
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
