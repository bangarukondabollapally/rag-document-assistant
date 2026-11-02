import { describe, expect, it } from "vitest";
import { conversationReducer as reduce, type Turn } from "../state/conversation";
import { SOURCES } from "./fixtures";

const start = (): Turn[] => reduce([], { type: "submit", id: 1, question: "Why?" });

describe("conversationReducer", () => {
  it("starts in retrieval, with no invented sources or answer", () => {
    const [turn] = start();
    expect(turn).toMatchObject({ stage: "retrieving", sources: null, answer: "", outcome: null });
  });

  it("accumulates tokens and records the final outcome", () => {
    let turns = start();
    turns = reduce(turns, { type: "event", id: 1, event: { type: "sources", sources: SOURCES, retrieval_ms: 9 } });
    turns = reduce(turns, { type: "event", id: 1, event: { type: "stage", stage: "generating" } });
    turns = reduce(turns, { type: "event", id: 1, event: { type: "token", text: "Hel" } });
    turns = reduce(turns, { type: "event", id: 1, event: { type: "token", text: "lo" } });
    turns = reduce(turns, {
      type: "event",
      id: 1,
      event: { type: "done", outcome: "answered", generation_ms: 5, model: "m" },
    });
    expect(turns[0]).toMatchObject({ answer: "Hello", stage: "complete", outcome: "answered", retrievalMs: 9, model: "m" });
  });

  it("keeps the partial answer and sources when generation fails", () => {
    let turns = start();
    turns = reduce(turns, { type: "event", id: 1, event: { type: "sources", sources: SOURCES, retrieval_ms: 9 } });
    turns = reduce(turns, { type: "event", id: 1, event: { type: "token", text: "Partial" } });
    turns = reduce(turns, {
      type: "event",
      id: 1,
      event: { type: "error", stage: "generating", code: "rate_limited", message: "Limit", retry_after_seconds: 5 },
    });
    expect(turns[0]).toMatchObject({ stage: "failed", answer: "Partial", sources: SOURCES });
    expect(turns[0]?.error?.retry_after_seconds).toBe(5);
  });

  it("treats a stream that ends without a result as a failure", () => {
    const turns = reduce(start(), { type: "stream_ended", id: 1 });
    expect(turns[0]?.stage).toBe("failed");
    expect(turns[0]?.error?.code).toBe("network");
  });

  it("does not overwrite a completed turn when the stream ends or stop arrives late", () => {
    let turns = reduce(start(), {
      type: "event",
      id: 1,
      event: { type: "done", outcome: "answered", generation_ms: 5, model: "m" },
    });
    turns = reduce(turns, { type: "stream_ended", id: 1 });
    turns = reduce(turns, { type: "stopped", id: 1 });
    expect(turns[0]?.stage).toBe("complete");
  });

  it("only updates the turn an event belongs to", () => {
    let turns = reduce(start(), { type: "submit", id: 2, question: "Second" });
    turns = reduce(turns, { type: "event", id: 2, event: { type: "token", text: "x" } });
    expect(turns[0]?.answer).toBe("");
    expect(turns[1]?.answer).toBe("x");
  });

  it("clears on reset, and ignores events for a cleared turn", () => {
    let turns = reduce(start(), { type: "reset" });
    turns = reduce(turns, { type: "event", id: 1, event: { type: "token", text: "late" } });
    expect(turns).toEqual([]);
  });
});
