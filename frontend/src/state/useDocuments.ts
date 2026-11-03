import { useCallback, useMemo, useState } from "react";
import { ApiError, deleteDocument, reindexDocument, uploadDocument } from "../api/client";
import type { DocumentSummary, Workspace } from "../api/types";

const SELECTION_KEY = "rag.selectedDocuments";

export type PendingAction = "deleting" | "reindexing";

export interface DocumentsState {
  /** Selected documents that are ready. Only these are ever sent with a question. */
  selected: DocumentSummary[];
  selectedIds: string[];
  /** With nothing selected, the document that explains why: it's indexing or it failed. */
  blocking: DocumentSummary | null;
  isChecked: (id: string) => boolean;
  toggle: (id: string) => void;
  uploads: string[];
  pending: Record<string, PendingAction>;
  error: string | null;
  upload: (file: File) => Promise<void>;
  remove: (id: string) => Promise<void>;
  reindex: (id: string) => Promise<void>;
}

function readStoredSelection(): string[] | null {
  try {
    const stored = JSON.parse(localStorage.getItem(SELECTION_KEY) ?? "null") as unknown;
    return Array.isArray(stored) ? stored.filter((id): id is string => typeof id === "string") : null;
  } catch {
    return null;
  }
}

function storeSelection(ids: string[]) {
  try {
    localStorage.setItem(SELECTION_KEY, JSON.stringify(ids));
  } catch {
    // Storage can be unavailable. The selection then lasts only for this page view.
  }
}

function messageOf(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

export function useDocuments(workspace: Workspace | null, refresh: () => Promise<void>): DocumentsState {
  // null means nothing has been chosen yet, so the first ready document is used.
  const [chosen, setChosen] = useState<string[] | null>(readStoredSelection);
  const [uploads, setUploads] = useState<string[]>([]);
  const [pending, setPending] = useState<Record<string, PendingAction>>({});
  const [error, setError] = useState<string | null>(null);

  const documents = useMemo(() => workspace?.documents ?? [], [workspace]);
  const ready = useMemo(() => documents.filter((document) => document.status === "ready"), [documents]);
  const wanted = useMemo(() => chosen ?? ready.slice(0, 1).map((document) => document.id), [chosen, ready]);
  const selected = useMemo(() => ready.filter((document) => wanted.includes(document.id)), [ready, wanted]);
  const selectedIds = useMemo(() => selected.map((document) => document.id), [selected]);
  const blocking = useMemo(() => {
    if (selected.length > 0) return null;
    const unavailable = documents.filter((document) => document.status !== "ready");
    const wantedUnavailable = unavailable.filter((document) => wanted.includes(document.id));
    const candidates = wantedUnavailable.length > 0 || ready.length > 0 ? wantedUnavailable : unavailable;
    return candidates.find((document) => document.status !== "failed") ?? candidates[0] ?? null;
  }, [documents, ready, selected, wanted]);

  const choose = useCallback((ids: string[]) => {
    setChosen(ids);
    storeSelection(ids);
  }, []);

  const toggle = useCallback(
    (id: string) => choose(wanted.includes(id) ? wanted.filter((other) => other !== id) : [...wanted, id]),
    [choose, wanted],
  );

  const isChecked = useCallback((id: string) => selectedIds.includes(id), [selectedIds]);

  const upload = useCallback(
    async (file: File) => {
      setError(null);
      const limit = workspace?.limits.max_upload_bytes;
      if (limit !== undefined && file.size > limit) {
        setError(`${file.name} is too large. PDFs can be up to ${Math.floor(limit / (1024 * 1024))} MB.`);
        return;
      }
      setUploads((current) => [...current, file.name]);
      try {
        const document = await uploadDocument(file);
        // Questions go to the document just uploaded, and only to it, so an earlier
        // selection can't answer for it or get mixed in with it.
        choose([document.id]);
        // Awaited so the "Uploading" row stays until the list includes the new document.
        await refresh();
      } catch (caught) {
        setError(messageOf(caught, `${file.name} couldn't be uploaded.`));
      } finally {
        setUploads((current) => {
          const index = current.indexOf(file.name);
          return index === -1 ? current : current.filter((_, position) => position !== index);
        });
      }
    },
    [choose, refresh, workspace],
  );

  const runAction = useCallback(
    async (id: string, action: PendingAction, run: () => Promise<void>, failure: string) => {
      setError(null);
      setPending((current) => ({ ...current, [id]: action }));
      try {
        await run();
      } catch (caught) {
        setError(messageOf(caught, failure));
      } finally {
        setPending(({ [id]: _finished, ...rest }) => rest);
        void refresh();
      }
    },
    [refresh],
  );

  const remove = useCallback(
    (id: string) => runAction(id, "deleting", () => deleteDocument(id), "That document couldn't be deleted."),
    [runAction],
  );

  const reindex = useCallback(
    (id: string) => runAction(id, "reindexing", () => reindexDocument(id), "That document couldn't be reindexed."),
    [runAction],
  );

  return { selected, selectedIds, blocking, isChecked, toggle, uploads, pending, error, upload, remove, reindex };
}
