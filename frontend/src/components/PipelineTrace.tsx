import type { Turn } from "../state/conversation";
import styles from "./PipelineTrace.module.css";

type StepState = "pending" | "active" | "done" | "failed" | "stopped";

function formatMs(ms: number): string {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function retrievalStep(turn: Turn): { state: StepState; text: string } {
  if (turn.sources) {
    const count = turn.sources.length;
    const time = turn.retrievalMs !== null ? ` in ${formatMs(turn.retrievalMs)}` : "";
    return { state: "done", text: `Retrieved ${count} ${count === 1 ? "passage" : "passages"}${time}` };
  }
  if (turn.stage === "failed") return { state: "failed", text: "Search failed" };
  if (turn.stage === "stopped") return { state: "stopped", text: "Search stopped" };
  return { state: "active", text: "Searching the document" };
}

function generationStep(turn: Turn): { state: StepState; text: string } {
  switch (turn.stage) {
    case "retrieving":
      return { state: "pending", text: "Generate answer" };
    case "generating":
      return { state: "active", text: "Generating answer" };
    case "complete": {
      const time = turn.generationMs !== null ? ` in ${formatMs(turn.generationMs)}` : "";
      return { state: "done", text: `${turn.model ?? "Model"} responded${time}` };
    }
    case "stopped":
      return { state: "stopped", text: turn.sources ? "Generation stopped" : "Not started" };
    case "failed":
      return turn.error?.stage === "generating"
        ? { state: "failed", text: "Generation failed" }
        : { state: "pending", text: "Not started" };
  }
}

export function PipelineTrace({ turn }: { turn: Turn }) {
  const steps = [retrievalStep(turn), generationStep(turn)];
  return (
    <ol className={styles.trace} aria-label="Pipeline progress">
      {steps.map((step) => (
        <li key={step.text} className={`${styles.step} ${styles[step.state]}`}>
          <span className={styles.marker} aria-hidden="true" />
          {step.text}
        </li>
      ))}
    </ol>
  );
}
