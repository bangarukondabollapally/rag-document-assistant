import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { answeredEvents, documentFixture, ndjsonResponse, SOURCES, stubApi, workspaceFixture } from "./fixtures";

// Loads the lazy answer chunk up front, so assertions don't race its import on a busy machine.
beforeAll(async () => {
  await import("../components/AnswerBlock");
});

async function ask(question = "What does it do?") {
  const user = userEvent.setup();
  const input = await screen.findByLabelText("Question about the document");
  await waitFor(() => expect(input).toBeEnabled());
  await user.type(input, `${question}{Enter}`);
  return user;
}

describe("App", () => {
  it("blocks asking while the saved index loads, and reports it", async () => {
    const workspace = workspaceFixture({ state: "loading", stage: "opening_indexes", elapsed_seconds: 3 });
    vi.stubGlobal("fetch", stubApi(workspace, () => ndjsonResponse([])));
    render(<App />);

    expect(await screen.findByText(/Loading the search index\. 3s elapsed/)).toBeInTheDocument();
    expect(screen.getByLabelText("Question about the document")).toBeDisabled();
    expect(screen.getByRole("button", { name: /What is an LSTM/ })).toBeDisabled();
  });

  it("only reads workspace status when it loads, and never requests a build", async () => {
    const fetchMock = vi.fn(stubApi(workspaceFixture(), () => ndjsonResponse([])));
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    await screen.findByText(/^Ready/);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(["/api/workspace"]);
  });

  it("omits the suggestions section for a document without suggested questions", async () => {
    const workspace = workspaceFixture({}, [documentFixture({ suggested_questions: [] })]);
    vi.stubGlobal("fetch", stubApi(workspace, () => ndjsonResponse([])));
    render(<App />);

    await screen.findByText(/Ask a question about/);
    expect(screen.queryByText("Try one")).not.toBeInTheDocument();
  });

  it("shows the error when the search index can't be opened", async () => {
    const workspace = workspaceFixture({ state: "failed", error: "The search index could not be opened." });
    vi.stubGlobal("fetch", stubApi(workspace, () => ndjsonResponse([])));
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("The search index could not be opened.");
  });

  it("explains how to start the API when it can't be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    render(<App />);

    expect(await screen.findByText("uvicorn api:app --port 8000")).toBeInTheDocument();
  });

  it("shows question, sources and answer, and links a passage to its excerpt", async () => {
    vi.stubGlobal("fetch", stubApi(workspaceFixture(), () => ndjsonResponse(answeredEvents())));
    render(<App />);
    const user = await ask();

    expect(await screen.findByRole("heading", { name: "What does it do?" })).toBeInTheDocument();
    expect(await screen.findByText("discards")).toBeInTheDocument();
    expect(screen.getByText("Retrieved 2 passages in 42 ms")).toBeInTheDocument();
    expect(screen.getByText("test-model responded in 1.2 s")).toBeInTheDocument();

    const panel = screen.getByRole("complementary", { name: "Retrieved context" });
    expect(within(panel).getByText(SOURCES[0]!.text)).toBeInTheDocument();

    const chip = screen.getByRole("button", { name: "Inspect passage 2, page 230" });
    await user.click(chip);
    const excerpt = within(panel).getByText(SOURCES[1]!.text);
    expect(excerpt.closest("li")).toHaveFocus();
  });

  it("presents the refusal as not found, with the closest passages", async () => {
    const events = answeredEvents("I couldn't find that in the document.").map((event) =>
      event.type === "done" ? { ...event, outcome: "not_found" as const } : event,
    );
    vi.stubGlobal("fetch", stubApi(workspaceFixture(), () => ndjsonResponse(events)));
    render(<App />);
    await ask();

    expect(await screen.findByText("Not found in this document")).toBeInTheDocument();
    expect(screen.getByText("Closest passages")).toBeInTheDocument();
    expect(screen.queryByText("I couldn't find that in the document.")).not.toBeInTheDocument();
  });

  it("keeps the partial answer on a rate limit, then retries the same question", async () => {
    const failed = [
      ...answeredEvents("Partial text").slice(0, 4),
      {
        type: "error" as const,
        stage: "generating" as const,
        code: "rate_limited" as const,
        message: "The limit was reached.",
        retry_after_seconds: 5,
      },
    ];
    const query = vi.fn(() => ndjsonResponse(failed));
    vi.stubGlobal("fetch", stubApi(workspaceFixture(), query));
    render(<App />);
    const user = await ask();

    expect(await screen.findByRole("alert")).toHaveTextContent("Try again in about 5 seconds");
    expect(screen.getByText("Partial text")).toBeInTheDocument();
    expect(screen.getByText("The answer above is incomplete.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(query).toHaveBeenCalledTimes(2));
  });

  it("does not render images or raw HTML from model output", async () => {
    const answer = 'Text ![x](https://evil.example/a.png) <img src="https://evil.example/b.png"> done';
    vi.stubGlobal("fetch", stubApi(workspaceFixture(), () => ndjsonResponse(answeredEvents(answer))));
    const { container } = render(<App />);
    await ask();

    await screen.findByText(/done/);
    expect(container.querySelector("img")).toBeNull();
  });

  it("renders the model's LaTeX as math", async () => {
    const answer = String.raw`The sigmoid is \(\sigma(z)\) here.`;
    vi.stubGlobal("fetch", stubApi(workspaceFixture(), () => ndjsonResponse(answeredEvents(answer))));
    const { container } = render(<App />);
    await ask();

    await waitFor(() => expect(container.querySelector(".katex")).not.toBeNull());
    expect(container.textContent).not.toContain(String.raw`\sigma`.concat("(z)\\)"));
  });

  it("keeps a list intact around display math, as real answers format it", async () => {
    const answer = [
      "1. **First step**",
      "   \\[",
      "   \\delta = \\frac{1}{2},",
      "   \\]",
      "   where \\(\\delta\\) is the *error signal*.",
      "2. **Second step**",
      "   Continue.",
    ].join("\n");
    vi.stubGlobal("fetch", stubApi(workspaceFixture(), () => ndjsonResponse(answeredEvents(answer))));
    const { container } = render(<App />);
    await ask();

    const signal = await screen.findByText("error signal");
    expect(signal.tagName).toBe("EM");
    expect(container.querySelector("article pre")).toBeNull();
    expect(container.querySelectorAll("[aria-busy] ol > li")).toHaveLength(2);
    expect(container.querySelectorAll(".katex")).toHaveLength(2);
    expect(container.querySelector("article")?.textContent).not.toContain("$$");
  });

  it("marks a stopped question as stopped", async () => {
    const hung = (signal: AbortSignal) =>
      new Promise<Response>((_, reject) => {
        signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      });
    vi.stubGlobal("fetch", (async (url: string, init?: RequestInit) =>
      url === "/api/workspace"
        ? new Response(JSON.stringify(workspaceFixture()))
        : hung(init!.signal!)) as unknown as typeof fetch);
    render(<App />);
    const user = await ask();

    await user.click(await screen.findByRole("button", { name: "Stop generating" }));
    expect(await screen.findByText("Stopped.")).toBeInTheDocument();
  });
});
