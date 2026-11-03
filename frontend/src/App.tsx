import { useCallback, useState } from "react";
import { AppShell } from "./components/AppShell";
import { Conversation } from "./components/Conversation";
import { QueryInput } from "./components/QueryInput";
import { Sidebar } from "./components/Sidebar";
import { SourcePanel, type SourceFocus } from "./components/SourcePanel";
import { useConversation } from "./state/useConversation";
import { useDocuments } from "./state/useDocuments";
import { useMediaQuery } from "./state/useMediaQuery";
import { useWorkspace } from "./state/useWorkspace";

export function App() {
  const { workspace, unreachable, refresh } = useWorkspace();
  const documents = useDocuments(workspace, refresh);
  const { turns, ask, stop, reset, busy } = useConversation();
  const [focus, setFocus] = useState<SourceFocus | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const isWide = useMediaQuery("(min-width: 1180px)");

  const panelTurn = turns.find((turn) => turn.id === focus?.turnId) ?? turns[turns.length - 1] ?? null;
  const { selected, selectedIds } = documents;
  const ready = workspace?.index.state === "ready" && !unreachable && selectedIds.length > 0;

  const handleAsk = useCallback(
    (question: string) => {
      setFocus(null);
      void ask(question, selectedIds);
    },
    [ask, selectedIds],
  );

  const handleReset = useCallback(() => {
    setFocus(null);
    setSourcesOpen(false);
    reset();
  }, [reset]);

  const inspectSource = useCallback((turnId: number, rank: number) => {
    setFocus({ turnId, rank });
    setSourcesOpen(true);
  }, []);

  let inputBlockedReason: string | null = null;
  if (unreachable) inputBlockedReason = "The API is unreachable.";
  else if (!workspace) inputBlockedReason = "Connecting to the API…";
  else if (workspace.index.state === "loading") inputBlockedReason = "Available when the index has loaded.";
  else if (workspace.index.state === "failed") inputBlockedReason = "The index couldn't be opened.";
  else if (workspace.documents.length === 0) inputBlockedReason = "Upload a PDF to get started.";
  else if (selected.length === 0) {
    const anyReady = workspace.documents.some((document) => document.status === "ready");
    const blocking = documents.blocking;
    if (blocking && blocking.status !== "failed") inputBlockedReason = "Available when a document is ready.";
    else if (anyReady) inputBlockedReason = "Select a document to ask a question.";
    else inputBlockedReason = "Upload a PDF that can be indexed to ask a question.";
  }

  const scope = selected.length === 1 ? (selected[0]?.name ?? null) : selected.length > 1 ? `${selected.length} documents` : null;

  return (
    <AppShell
      documentName={scope}
      hasTurns={turns.length > 0}
      sourceCount={panelTurn?.sources?.length ?? 0}
      sourcesOpen={sourcesOpen}
      onSourcesOpenChange={setSourcesOpen}
      onReset={handleReset}
      sidebar={<Sidebar workspace={workspace} unreachable={unreachable} documents={documents} />}
      sources={
        <SourcePanel
          turn={panelTurn}
          pipeline={workspace?.pipeline ?? null}
          focus={focus}
          onClose={isWide ? undefined : () => setSourcesOpen(false)}
        />
      }
      conversation={
        <Conversation
          workspace={workspace}
          unreachable={unreachable}
          turns={turns}
          ready={ready}
          documents={documents}
          onAsk={handleAsk}
          onInspectSource={inspectSource}
        />
      }
      input={
        <QueryInput
          busy={busy}
          blockedReason={inputBlockedReason}
          conversationMemory={workspace?.capabilities.conversation_memory ?? null}
          onSubmit={handleAsk}
          onStop={stop}
        />
      }
    />
  );
}
