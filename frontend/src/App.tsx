import { useCallback, useState } from "react";
import { AppShell } from "./components/AppShell";
import { Conversation } from "./components/Conversation";
import { QueryInput } from "./components/QueryInput";
import { Sidebar } from "./components/Sidebar";
import { SourcePanel, type SourceFocus } from "./components/SourcePanel";
import { useConversation } from "./state/useConversation";
import { useMediaQuery } from "./state/useMediaQuery";
import { useWorkspace } from "./state/useWorkspace";

export function App() {
  const { workspace, unreachable } = useWorkspace();
  const { turns, ask, stop, reset, busy } = useConversation();
  const [focus, setFocus] = useState<SourceFocus | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const isWide = useMediaQuery("(min-width: 1180px)");

  const panelTurn = turns.find((turn) => turn.id === focus?.turnId) ?? turns[turns.length - 1] ?? null;
  const ready = workspace?.index.state === "ready" && !unreachable;

  const handleAsk = useCallback(
    (question: string) => {
      setFocus(null);
      void ask(question);
    },
    [ask],
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
  else if (workspace.index.state === "not_indexed") inputBlockedReason = "This document hasn't been indexed.";
  else if (workspace.index.state === "failed") inputBlockedReason = "The index couldn't be opened.";

  return (
    <AppShell
      documentName={workspace?.documents[0]?.name ?? null}
      hasTurns={turns.length > 0}
      sourceCount={panelTurn?.sources?.length ?? 0}
      sourcesOpen={sourcesOpen}
      onSourcesOpenChange={setSourcesOpen}
      onReset={handleReset}
      sidebar={<Sidebar workspace={workspace} unreachable={unreachable} />}
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
