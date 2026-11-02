import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import type { Source } from "../api/types";
import { answeredEvents, documentFixture, fakeApi, jsonResponse, ndjsonResponse, workspaceFixture } from "./fixtures";

beforeAll(async () => {
  await import("../components/AnswerBlock");
});

const QUESTION_INPUT = "Question about the document";
const documentA = documentFixture({ id: "doc-a", name: "a.pdf", title: null, chunks: 100, pages_indexed: 10, suggested_questions: [] });
const documentB = documentFixture({ id: "doc-b", name: "b.pdf", title: null, chunks: 200, pages_indexed: 20, suggested_questions: [] });

function pdfFile(name = "new.pdf", size = 10) {
  return new File(["x".repeat(size)], name, { type: "application/pdf" });
}

function start(api: ReturnType<typeof fakeApi>) {
  vi.stubGlobal("fetch", api.fetch);
  const user = userEvent.setup();
  const view = render(<App />);
  return { user, ...view };
}

function checkbox(name: string) {
  return screen.findByRole("checkbox", { name });
}

describe("document list", () => {
  it("shows the real name, status, pages and chunks of each document", async () => {
    start(fakeApi(workspaceFixture({}, [documentA, documentB])));
    expect(await checkbox("a.pdf")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "b.pdf" })).toBeInTheDocument();
    expect(screen.getByText("Ready · 10 pages · 100 chunks")).toBeInTheDocument();
    expect(screen.getByText("Ready · 20 pages · 200 chunks")).toBeInTheDocument();
  });

  it("shows the empty state when there are no documents", async () => {
    start(fakeApi(workspaceFixture({}, [])));
    expect(await screen.findByRole("heading", { name: "No documents yet" })).toBeInTheDocument();
    expect(screen.getAllByText("Upload a PDF to get started").length).toBeGreaterThan(0);
    expect(screen.getByLabelText(QUESTION_INPUT)).toBeDisabled();
    expect(screen.getByLabelText(QUESTION_INPUT)).toHaveAttribute("placeholder", "Upload a PDF to get started.");
  });

  it("shows a queued document truthfully, with no percentage", async () => {
    start(fakeApi(workspaceFixture({}, [documentFixture({ status: "uploaded", chunks: null })])));
    expect(await screen.findByText("Waiting to index")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/\d+\s*%/);
  });

  it("shows an indexing document without fake progress, and blocks selecting it", async () => {
    start(fakeApi(workspaceFixture({}, [documentFixture({ status: "indexing", chunks: null })])));
    expect(await screen.findByText("Indexing…")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/\d+\s*%/);
    expect(screen.getByRole("checkbox", { name: "sample.pdf" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
    expect(screen.getByLabelText(QUESTION_INPUT)).toHaveAttribute("placeholder", "Available when a document is ready.");
  });

  it("shows a failed document's reason and offers a retry", async () => {
    const api = fakeApi(
      workspaceFixture({}, [documentFixture({ id: "doc-f", status: "failed", error: "scan.pdf has no extractable text." })]),
      { "POST /api/documents/doc-f/reindex": () => jsonResponse({}, 202) },
    );
    const { user } = start(api);
    expect(await screen.findByText("scan.pdf has no extractable text.")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "sample.pdf" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Retry indexing" }));
    await waitFor(() => expect(api.calls.map((call) => call.url)).toContain("/api/documents/doc-f/reindex"));
  });

  it("polls while a document is indexing, and shows it ready afterwards", async () => {
    const api = fakeApi(workspaceFixture({}, [documentFixture({ status: "indexing" })]));
    start(api);
    await screen.findByText("Indexing…");
    api.setWorkspace(workspaceFixture({}, [documentFixture({ status: "ready", pages_indexed: 3, chunks: 7 })]));
    expect(await screen.findByText("Ready · 3 pages · 7 chunks", {}, { timeout: 3000 })).toBeInTheDocument();
  });

  it("reports that the search index is loading", async () => {
    start(fakeApi(workspaceFixture({ state: "loading", stage: "loading_model", elapsed_seconds: 3 })));
    expect(await screen.findByText(/Loading the search index\. 3s elapsed/)).toBeInTheDocument();
    expect(screen.getByLabelText(QUESTION_INPUT)).toBeDisabled();
  });
});

describe("upload", () => {
  function uploadInput(container: HTMLElement) {
    return container.querySelector<HTMLInputElement>('input[type="file"]')!;
  }

  it("shows a truthful uploading state, then the document from the API", async () => {
    let finish: (response: Response) => void = () => {};
    const created = documentFixture({ id: "doc-new", name: "new.pdf", status: "uploaded" });
    const api = fakeApi(workspaceFixture({}, [documentA]), {
      "POST /api/documents": () => new Promise<Response>((resolve) => (finish = resolve)),
    });
    const { user, container } = start(api);
    await checkbox("a.pdf");

    await user.upload(uploadInput(container), pdfFile());
    expect(await screen.findByText("Uploading…")).toBeInTheDocument();
    expect(screen.getByText("new.pdf")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/\d+\s*%/);

    api.setWorkspace(workspaceFixture({}, [documentA, created]));
    finish(jsonResponse(created, 201));
    expect(await screen.findByText("Waiting to index")).toBeInTheDocument();
    expect(screen.queryByText("Uploading…")).not.toBeInTheDocument();
    expect(api.calls[0]?.body).toBeInstanceOf(FormData);
  });

  it("shows the server's reason when a duplicate is rejected", async () => {
    const api = fakeApi(workspaceFixture({}, [documentA]), {
      "POST /api/documents": () =>
        jsonResponse({ code: "duplicate", message: "This PDF is already uploaded as a.pdf.", document_id: "doc-a" }, 409),
    });
    const { user, container } = start(api);
    await checkbox("a.pdf");
    await user.upload(uploadInput(container), pdfFile());
    expect(await screen.findByRole("alert")).toHaveTextContent("already uploaded as a.pdf");
    expect(screen.queryByText("Uploading…")).not.toBeInTheDocument();
  });

  it("rejects an oversized file without sending it", async () => {
    const workspace = { ...workspaceFixture({}, [documentA]), limits: { max_upload_bytes: 5 } };
    const api = fakeApi(workspace);
    const { user, container } = start(api);
    await checkbox("a.pdf");
    await user.upload(uploadInput(container), pdfFile("big.pdf", 100));
    expect(await screen.findByRole("alert")).toHaveTextContent("big.pdf is too large");
    expect(api.calls).toEqual([]);
  });

  it("offers upload from the empty state, and selects the first document automatically", async () => {
    const created = documentFixture({ id: "doc-new", name: "new.pdf", status: "uploaded" });
    const api = fakeApi(workspaceFixture({}, []), {
      "POST /api/documents": () => jsonResponse(created, 201),
    });
    const { user, container } = start(api);
    await screen.findByRole("heading", { name: "No documents yet" });
    api.setWorkspace(workspaceFixture({}, [created]));
    await user.upload(uploadInput(container), pdfFile());
    await screen.findByText("Waiting to index");

    api.setWorkspace(workspaceFixture({}, [documentFixture({ ...created, status: "ready" })]));
    expect(await checkbox("new.pdf")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("checkbox", { name: "new.pdf" })).toBeChecked(), { timeout: 3000 });
  });
});

describe("selection", () => {
  it("selects only the first ready document by default", async () => {
    start(fakeApi(workspaceFixture({}, [documentA, documentB])));
    expect(await checkbox("a.pdf")).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "b.pdf" })).not.toBeChecked();
    expect(screen.getByRole("heading", { name: "Ask a question about a.pdf" })).toBeInTheDocument();
  });

  it("names the scope in the header and heading as the selection changes", async () => {
    const { user } = start(fakeApi(workspaceFixture({}, [documentA, documentB])));
    await user.click(await checkbox("b.pdf"));
    expect(screen.getByRole("heading", { name: "Ask a question about 2 documents" })).toBeInTheDocument();
    expect(screen.getByText("2 documents")).toBeInTheDocument();
  });

  it("blocks asking when nothing is selected, instead of searching everything", async () => {
    const api = fakeApi(workspaceFixture({}, [documentA, documentB]));
    const { user } = start(api);
    await user.click(await checkbox("a.pdf"));

    expect(screen.getByRole("heading", { name: "Select a document" })).toBeInTheDocument();
    const input = screen.getByLabelText(QUESTION_INPUT);
    expect(input).toBeDisabled();
    expect(input).toHaveAttribute("placeholder", "Select a document to ask a question.");
    expect(api.calls).toEqual([]);
  });

  it("remembers the selection across reloads", async () => {
    const first = start(fakeApi(workspaceFixture({}, [documentA, documentB])));
    await first.user.click(await checkbox("a.pdf"));
    await first.user.click(await checkbox("b.pdf"));
    first.unmount();

    start(fakeApi(workspaceFixture({}, [documentA, documentB])));
    expect(await checkbox("b.pdf")).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "a.pdf" })).not.toBeChecked();
  });

  it("ignores a remembered document that no longer exists", async () => {
    localStorage.setItem("rag.selectedDocuments", JSON.stringify(["deleted-id"]));
    start(fakeApi(workspaceFixture({}, [documentA])));
    expect(await checkbox("a.pdf")).not.toBeChecked();
    expect(screen.getByRole("heading", { name: "Select a document" })).toBeInTheDocument();
  });

  it("survives unreadable saved selection data", async () => {
    localStorage.setItem("rag.selectedDocuments", "not json");
    start(fakeApi(workspaceFixture({}, [documentA])));
    expect(await checkbox("a.pdf")).toBeChecked();
  });

  it("drops a selected document from the scope while it isn't ready", async () => {
    start(fakeApi(workspaceFixture({}, [documentFixture({ ...documentA, status: "indexing" }), documentB])));
    expect(await checkbox("a.pdf")).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "a.pdf" })).toBeDisabled();
  });
});

describe("delete and reindex", () => {
  it("asks for confirmation, shows progress, then removes the document", async () => {
    let finish: (response: Response) => void = () => {};
    const api = fakeApi(workspaceFixture({}, [documentA, documentB]), {
      "DELETE /api/documents/doc-b": () => new Promise<Response>((resolve) => (finish = resolve)),
    });
    const { user } = start(api);
    await checkbox("b.pdf");
    const item = screen.getByRole("checkbox", { name: "b.pdf" }).closest("li")!;

    await user.click(within(item).getByRole("button", { name: "Delete" }));
    expect(api.calls).toEqual([]);
    await user.click(within(item).getByRole("button", { name: "Yes, delete" }));
    expect(await within(item).findByText("Deleting…")).toBeInTheDocument();

    api.setWorkspace(workspaceFixture({}, [documentA]));
    finish(new Response(null, { status: 204 }));
    await waitFor(() => expect(screen.queryByRole("checkbox", { name: "b.pdf" })).not.toBeInTheDocument());
    expect(screen.getByRole("checkbox", { name: "a.pdf" })).toBeInTheDocument();
  });

  it("can cancel a delete", async () => {
    const api = fakeApi(workspaceFixture({}, [documentA]));
    const { user } = start(api);
    await checkbox("a.pdf");
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(api.calls).toEqual([]);
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();
  });

  it("shows why a delete failed", async () => {
    const api = fakeApi(workspaceFixture({}, [documentA]), {
      "DELETE /api/documents/doc-a": () => jsonResponse({ code: "document_busy", message: "It's being indexed." }, 409),
    });
    const { user } = start(api);
    await checkbox("a.pdf");
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await user.click(screen.getByRole("button", { name: "Yes, delete" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("It's being indexed.");
  });

  it("confirms before reindexing, and only requests that document", async () => {
    const api = fakeApi(workspaceFixture({}, [documentA, documentB]), {
      "POST /api/documents/doc-a/reindex": () => jsonResponse({}, 202),
    });
    const { user } = start(api);
    await checkbox("a.pdf");
    const item = screen.getByRole("checkbox", { name: "a.pdf" }).closest("li")!;
    await user.click(within(item).getByRole("button", { name: "Reindex" }));
    expect(api.calls).toEqual([]);
    await user.click(within(item).getByRole("button", { name: "Yes, reindex" }));
    await waitFor(() => expect(api.calls.map((call) => call.url)).toEqual(["/api/documents/doc-a/reindex"]));
  });
});

describe("question flow", () => {
  async function askQuestion(user: ReturnType<typeof userEvent.setup>) {
    const input = await screen.findByLabelText(QUESTION_INPUT);
    await waitFor(() => expect(input).toBeEnabled());
    await user.type(input, "What is it?{Enter}");
  }

  it("sends exactly the selected documents with the question", async () => {
    const api = fakeApi(workspaceFixture({}, [documentA, documentB, documentFixture({ id: "doc-c", name: "c.pdf" })]), {
      "POST /api/query": () => ndjsonResponse(answeredEvents()),
    });
    const { user } = start(api);
    await user.click(await checkbox("b.pdf"));
    await askQuestion(user);

    await waitFor(() => expect(api.calls.some((call) => call.url === "/api/query")).toBe(true));
    const call = api.calls.find((item) => item.url === "/api/query");
    expect(call?.body).toEqual({ question: "What is it?", selected_document_ids: ["doc-a", "doc-b"] });
  });

  it("shows a server refusal when a document stops being ready", async () => {
    const api = fakeApi(workspaceFixture({}, [documentA]), {
      "POST /api/query": () => jsonResponse({ code: "document_not_ready", message: "a.pdf isn't ready yet." }, 409),
    });
    const { user } = start(api);
    await askQuestion(user);
    expect(await screen.findByRole("alert")).toHaveTextContent("a.pdf isn't ready yet.");
  });

  it("uses plural wording when several documents don't contain the answer", async () => {
    const events = answeredEvents("I couldn't find that in the document.").map((event) =>
      event.type === "done" ? { ...event, outcome: "not_found" as const } : event,
    );
    const api = fakeApi(workspaceFixture({}, [documentA, documentB]), { "POST /api/query": () => ndjsonResponse(events) });
    const { user } = start(api);
    await user.click(await checkbox("b.pdf"));
    await askQuestion(user);
    expect(await screen.findByText("Not found in the selected documents")).toBeInTheDocument();
  });
});

describe("source rendering", () => {
  const mixed: Source[] = [
    { rank: 1, document_id: "doc-a", document: "a.pdf", page: 3, distance: 0.41, text: "From the first document." },
    { rank: 2, document_id: "doc-b", document: "b.pdf", page: 9, distance: 0.52, text: "From the second document." },
  ];

  async function askAcrossDocuments(sources: Source[]) {
    const api = fakeApi(workspaceFixture({}, [documentA, documentB]), {
      "POST /api/query": () => ndjsonResponse(answeredEvents("An answer.", sources)),
    });
    const { user } = start(api);
    await user.click(await checkbox("b.pdf"));
    const input = screen.getByLabelText(QUESTION_INPUT);
    await user.type(input, "What?{Enter}");
    await screen.findByText("An answer.");
    return user;
  }

  it("shows each source's own document, page, distance and text from the API", async () => {
    await askAcrossDocuments(mixed);
    const panel = screen.getByRole("complementary", { name: "Retrieved context" });
    expect(within(panel).getByText("a.pdf, p. 3")).toBeInTheDocument();
    expect(within(panel).getByText("b.pdf, p. 9")).toBeInTheDocument();
    expect(within(panel).getByText("0.41")).toBeInTheDocument();
    expect(within(panel).getByText("From the second document.")).toBeInTheDocument();
  });

  it("names the document on each passage button when sources span documents", async () => {
    const user = await askAcrossDocuments(mixed);
    const chip = screen.getByRole("button", { name: "Inspect passage 2, b.pdf, page 9" });
    await user.click(chip);
    const panel = screen.getByRole("complementary", { name: "Retrieved context" });
    expect(within(panel).getByText("From the second document.").closest("li")).toHaveFocus();
  });

  it("keeps the shorter passage button when sources come from one document", async () => {
    await askAcrossDocuments(mixed.map((source) => ({ ...source, document_id: "doc-a", document: "a.pdf" })));
    expect(screen.getByRole("button", { name: "Inspect passage 2, page 9" })).toBeInTheDocument();
  });

  it("shows no invented citations when no sources are returned", async () => {
    await askAcrossDocuments([]);
    expect(screen.queryByRole("button", { name: /Inspect passage/ })).not.toBeInTheDocument();
    expect(screen.queryByText("Context sent to the model")).not.toBeInTheDocument();
  });
});
