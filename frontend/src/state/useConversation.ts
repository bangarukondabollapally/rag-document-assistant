import { useCallback, useEffect, useReducer, useRef } from "react";
import { ApiError, streamQuery } from "../api/client";
import { conversationReducer, isBusy } from "./conversation";

export function useConversation() {
  const [turns, dispatch] = useReducer(conversationReducer, []);
  const nextId = useRef(1);
  const active = useRef<AbortController | null>(null);

  useEffect(() => () => active.current?.abort(), []);

  const ask = useCallback(async (question: string) => {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    const id = nextId.current++;
    dispatch({ type: "submit", id, question });

    try {
      for await (const event of streamQuery(question, controller.signal)) {
        dispatch({ type: "event", id, event });
      }
      dispatch({ type: "stream_ended", id });
    } catch (error) {
      if (controller.signal.aborted) {
        dispatch({ type: "stopped", id });
      } else if (error instanceof ApiError) {
        dispatch({ type: "failed", id, error: error.toQueryError() });
      } else {
        dispatch({
          type: "failed",
          id,
          error: { code: "network", message: "The connection to the API failed." },
        });
      }
    }
  }, []);

  const stop = useCallback(() => active.current?.abort(), []);

  const reset = useCallback(() => {
    active.current?.abort();
    dispatch({ type: "reset" });
  }, []);

  const last = turns[turns.length - 1];
  return { turns, ask, stop, reset, busy: last ? isBusy(last) : false };
}
