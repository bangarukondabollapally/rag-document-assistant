import { useEffect, useRef } from "react";
import type { PipelineConfig, Source } from "../api/types";
import type { Turn } from "../state/conversation";
import { formatExcerpt } from "./formatExcerpt";
import { CloseIcon } from "./Icons";
import styles from "./SourcePanel.module.css";

export interface SourceFocus {
  turnId: number;
  rank: number;
}

function SourceCard({ source, focus }: { source: Source; focus: SourceFocus | null }) {
  const ref = useRef<HTMLLIElement>(null);

  // Each click creates a new focus object, so re-clicking a passage scrolls to it again.
  useEffect(() => {
    if (focus) {
      ref.current?.scrollIntoView({ block: "nearest" });
      ref.current?.focus({ preventScroll: true });
    }
  }, [focus]);

  return (
    <li ref={ref} tabIndex={-1} className={`${styles.card} ${focus ? styles.focused : ""}`}>
      <div className={styles.cardHeader}>
        <span className={styles.rank}>{source.rank}</span>
        <span className={styles.origin}>
          {source.document}
          {source.page ? `, p. ${source.page}` : ""}
        </span>
        <span className={styles.distance} title="Squared L2 distance. Lower is closer.">
          {source.distance.toFixed(2)}
        </span>
      </div>
      <p className={styles.excerpt}>{formatExcerpt(source.text)}</p>
    </li>
  );
}

function RetrievalDetails({ turn, pipeline }: { turn: Turn; pipeline: PipelineConfig }) {
  return (
    <details className={styles.details}>
      <summary>How these were retrieved</summary>
      <div className={styles.detailsBody}>
        <p>
          {pipeline.search_type} picked {turn.sources?.length} of the {pipeline.fetch_k} nearest chunks (λ{" "}
          {pipeline.lambda}), balancing closeness to the question against repetition. They're listed in selection order.
        </p>
        <p>
          The number on each passage is its squared L2 distance from the question. Lower is closer. It measures
          similarity, so it can't show that a passage is correct.
        </p>
        <p>Page numbers are PDF pages, which can differ from the numbers printed in the document.</p>
        {turn.retrievalMs !== null && <p>Search time: {turn.retrievalMs} ms.</p>}
      </div>
    </details>
  );
}

interface SourcePanelProps {
  turn: Turn | null;
  pipeline: PipelineConfig | null;
  focus: SourceFocus | null;
  onClose?: () => void;
}

export function SourcePanel({ turn, pipeline, focus, onClose }: SourcePanelProps) {
  const sources = turn?.sources ?? null;

  return (
    <div className={styles.panel}>
      <header className={styles.header}>
        <h2 className={styles.title}>Retrieved context</h2>
        {onClose && (
          <button type="button" className={styles.close} aria-label="Close retrieved context" onClick={onClose}>
            <CloseIcon />
          </button>
        )}
      </header>

      <div className={styles.body}>
        {!turn && (
          <p className={styles.placeholder}>
            When you ask a question, the passages sent to the model appear here, with their pages.
          </p>
        )}
        {turn && !sources && (
          <p className={styles.placeholder}>{turn.error ? "No passages were retrieved." : "Searching the document…"}</p>
        )}
        {turn && sources && (
          <>
            <p className={styles.forQuestion}>{turn.question}</p>
            {pipeline && <RetrievalDetails turn={turn} pipeline={pipeline} />}
            <ol className={styles.list}>
              {sources.map((source) => (
                <SourceCard
                  key={source.rank}
                  source={source}
                  focus={focus?.turnId === turn.id && focus.rank === source.rank ? focus : null}
                />
              ))}
            </ol>
          </>
        )}
      </div>
    </div>
  );
}
