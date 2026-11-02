import { useEffect, useState } from "react";
import { fetchWorkspace } from "../api/client";
import type { Workspace } from "../api/types";

const LOADING_POLL_MS = 500;
const OFFLINE_POLL_MS = 2000;

export interface WorkspaceResult {
  workspace: Workspace | null;
  unreachable: boolean;
}

/** Polls only while the answer can change: while loading or when the API is unreachable. */
export function useWorkspace(): WorkspaceResult {
  const [result, setResult] = useState<WorkspaceResult>({ workspace: null, unreachable: false });

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function load() {
      try {
        const workspace = await fetchWorkspace(controller.signal);
        setResult({ workspace, unreachable: false });
        if (workspace.index.state === "loading") {
          timer = setTimeout(load, LOADING_POLL_MS);
        }
      } catch {
        if (controller.signal.aborted) return;
        setResult((current) => ({ ...current, unreachable: true }));
        timer = setTimeout(load, OFFLINE_POLL_MS);
      }
    }

    void load();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  return result;
}
