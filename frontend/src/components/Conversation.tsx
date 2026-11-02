import { useEffect, useRef } from "react";
import type { Workspace } from "../api/types";
import type { Turn } from "../state/conversation";
import { isBusy } from "../state/conversation";
import { ChatMessage } from "./ChatMessage";
import { EmptyState } from "./EmptyState";
import styles from "./Conversation.module.css";

const STAGE_ANNOUNCEMENT: Record<Turn["stage"], string> = {
  retrieving: "Searching the document",
  generating: "Generating the answer",
  complete: "Answer ready",
  stopped: "Stopped",
  failed: "The question failed",
};

interface ConversationProps {
  workspace: Workspace | null;
  unreachable: boolean;
  turns: Turn[];
  ready: boolean;
  onAsk: (question: string) => void;
  onInspectSource: (turnId: number, rank: number) => void;
}

export function Conversation({ workspace, unreachable, turns, ready, onAsk, onInspectSource }: ConversationProps) {
  const latest = useRef<HTMLElement>(null);
  const last = turns[turns.length - 1];
  const lastId = last?.id;

  // Bring each new question to the top, then leave the scroll alone so the reader
  // isn't pulled away while a long answer streams.
  useEffect(() => {
    if (lastId !== undefined) latest.current?.scrollIntoView({ block: "start" });
  }, [lastId]);

  if (turns.length === 0) {
    return <EmptyState workspace={workspace} unreachable={unreachable} ready={ready} onAsk={onAsk} />;
  }

  return (
    <div className={styles.conversation}>
      {turns.map((turn) => (
        <ChatMessage
          key={turn.id}
          ref={turn.id === lastId ? latest : undefined}
          turn={turn}
          busy={isBusy(turn)}
          onRetry={onAsk}
          onInspectSource={onInspectSource}
        />
      ))}
      <div role="status" className="sr-only">
        {last ? STAGE_ANNOUNCEMENT[last.stage] : ""}
      </div>
    </div>
  );
}
