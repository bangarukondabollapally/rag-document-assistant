import { useCallback, useEffect, useRef, useState } from "react";
import { fetchWorkspace } from "../api/client";
import type { Workspace } from "../api/types";

const LOADING_POLL_MS = 500;
const INDEXING_POLL_MS = 1000;
const OFFLINE_POLL_MS = 2000;

export interface WorkspaceResult {
  workspace: Workspace | null;
  unreachable: boolean;
  refresh: () => Promise<void>;
}

function nextPollDelay(workspace: Workspace): number | null {
  if (workspace.index.state === "loading") return LOADING_POLL_MS;
  const processing = workspace.documents.some((document) => document.status === "uploaded" || document.status === "indexing");
  return processing ? INDEXING_POLL_MS : null;
}

/** Polls only while the answer can change: while loading, indexing, or unreachable. */
export function useWorkspace(): WorkspaceResult {
  const [result, setResult] = useState<Omit<WorkspaceResult, "refresh">>({ workspace: null, unreachable: false });
  const reload = useRef<() => Promise<void>>(() => Promise.resolve());

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let latestRequest = 0;

    async function load() {
      clearTimeout(timer);
      const request = ++latestRequest;
      try {
        const workspace = await fetchWorkspace(controller.signal);
        // An older response must not overwrite a newer one, such as after a delete.
        if (request !== latestRequest) return;
        setResult({ workspace, unreachable: false });
        const delay = nextPollDelay(workspace);
        if (delay !== null) timer = setTimeout(load, delay);
      } catch {
        if (controller.signal.aborted || request !== latestRequest) return;
        setResult((current) => ({ ...current, unreachable: true }));
        timer = setTimeout(load, OFFLINE_POLL_MS);
      }
    }

    reload.current = load;
    void load();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  const refresh = useCallback(() => reload.current(), []);
  return { ...result, refresh };
}
