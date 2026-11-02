import { lazy, Suspense, type Ref } from "react";
import type { Turn } from "../state/conversation";
import { describeQueryError, ErrorState } from "./ErrorState";
import { PipelineTrace } from "./PipelineTrace";
import styles from "./ChatMessage.module.css";

// Markdown and math rendering aren't needed until the first answer arrives.
const AnswerBlock = lazy(() => import("./AnswerBlock"));

interface ChatMessageProps {
  ref?: Ref<HTMLElement>;
  turn: Turn;
  busy: boolean;
  onRetry: (question: string) => void;
  onInspectSource: (turnId: number, rank: number) => void;
}

export function ChatMessage({ ref, turn, busy, onRetry, onInspectSource }: ChatMessageProps) {
  const notFound = turn.outcome === "not_found";
  const hasSources = turn.sources !== null && turn.sources.length > 0;

  return (
    <article ref={ref} className={styles.turn}>
      <h2 className={styles.question}>{turn.question}</h2>
      <PipelineTrace turn={turn} />

      {hasSources && (
        <div className={styles.sources}>
          <span className={styles.sourcesLabel}>{notFound ? "Closest passages" : "Context sent to the model"}</span>
          <ul className={styles.chips}>
            {turn.sources?.map((source) => (
              <li key={source.rank}>
                <button
                  type="button"
                  className={styles.chip}
                  aria-label={`Inspect passage ${source.rank}${source.page ? `, page ${source.page}` : ""}`}
                  onClick={() => onInspectSource(turn.id, source.rank)}
                >
                  <span className={styles.chipRank}>{source.rank}</span>
                  {source.page ? `p. ${source.page}` : "no page"}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {notFound ? (
        <div className={styles.notFound}>
          <p className={styles.notFoundTitle}>Not found in this document</p>
          <p>
            The model reported that the {turn.sources?.length} closest passages don't answer this. You can inspect them
            to check, or try different wording.
          </p>
        </div>
      ) : (
        turn.answer && (
          <Suspense fallback={<p className={styles.plainAnswer}>{turn.answer}</p>}>
            <AnswerBlock text={turn.answer} streaming={busy} />
          </Suspense>
        )
      )}

      {turn.stage === "stopped" && <p className={styles.stopped}>Stopped{turn.answer ? " before the answer finished." : "."}</p>}

      {turn.error && (
        <div className={styles.error}>
          {turn.answer && <p className={styles.partial}>The answer above is incomplete.</p>}
          <ErrorState {...describeQueryError(turn.error)} onRetry={() => onRetry(turn.question)} />
        </div>
      )}
    </article>
  );
}
