import type { QueryError, QueryEvent, Source } from "../api/types";

export type TurnStage = "retrieving" | "generating" | "complete" | "stopped" | "failed";

export interface Turn {
  id: number;
  question: string;
  stage: TurnStage;
  sources: Source[] | null;
  answer: string;
  outcome: "answered" | "not_found" | null;
  retrievalMs: number | null;
  generationMs: number | null;
  model: string | null;
  error: (QueryError & { stage: "retrieving" | "generating" }) | null;
}

export type ConversationAction =
  | { type: "submit"; id: number; question: string }
  | { type: "event"; id: number; event: QueryEvent }
  | { type: "stopped"; id: number }
  | { type: "failed"; id: number; error: QueryError }
  | { type: "stream_ended"; id: number }
  | { type: "reset" };

export function isBusy(turn: Turn): boolean {
  return turn.stage === "retrieving" || turn.stage === "generating";
}

function applyEvent(turn: Turn, event: QueryEvent): Turn {
  switch (event.type) {
    case "stage":
      return { ...turn, stage: event.stage };
    case "sources":
      return { ...turn, sources: event.sources, retrievalMs: event.retrieval_ms };
    case "token":
      return { ...turn, answer: turn.answer + event.text };
    case "done":
      return {
        ...turn,
        stage: "complete",
        outcome: event.outcome,
        generationMs: event.generation_ms,
        model: event.model,
      };
    case "error": {
      const { type: _type, ...error } = event;
      return { ...turn, stage: "failed", error };
    }
  }
}

function failedStage(turn: Turn): "retrieving" | "generating" {
  return turn.stage === "generating" ? "generating" : "retrieving";
}

function updateTurn(turns: Turn[], id: number, update: (turn: Turn) => Turn): Turn[] {
  return turns.map((turn) => (turn.id === id ? update(turn) : turn));
}

export function conversationReducer(turns: Turn[], action: ConversationAction): Turn[] {
  switch (action.type) {
    case "submit":
      return [
        ...turns,
        {
          id: action.id,
          question: action.question,
          stage: "retrieving",
          sources: null,
          answer: "",
          outcome: null,
          retrievalMs: null,
          generationMs: null,
          model: null,
          error: null,
        },
      ];
    case "event":
      return updateTurn(turns, action.id, (turn) => applyEvent(turn, action.event));
    case "stopped":
      return updateTurn(turns, action.id, (turn) => (isBusy(turn) ? { ...turn, stage: "stopped" } : turn));
    case "failed":
      return updateTurn(turns, action.id, (turn) => ({
        ...turn,
        stage: "failed",
        error: { ...action.error, stage: failedStage(turn) },
      }));
    case "stream_ended":
      return updateTurn(turns, action.id, (turn) =>
        isBusy(turn)
          ? {
              ...turn,
              stage: "failed",
              error: {
                code: "network",
                message: "The connection closed before the answer finished.",
                stage: failedStage(turn),
              },
            }
          : turn,
      );
    case "reset":
      return [];
  }
}
