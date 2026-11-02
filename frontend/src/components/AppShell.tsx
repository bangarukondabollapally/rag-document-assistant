import { useState, type ReactNode } from "react";
import { useMediaQuery } from "../state/useMediaQuery";
import { CloseIcon, MenuIcon, PassagesIcon } from "./Icons";
import { Sheet } from "./Sheet";
import styles from "./AppShell.module.css";

interface AppShellProps {
  documentName: string | null;
  hasTurns: boolean;
  sourceCount: number;
  sourcesOpen: boolean;
  onSourcesOpenChange: (open: boolean) => void;
  onReset: () => void;
  sidebar: ReactNode;
  conversation: ReactNode;
  input: ReactNode;
  sources: ReactNode;
}

export function AppShell({
  documentName,
  hasTurns,
  sourceCount,
  sourcesOpen,
  onSourcesOpenChange,
  onReset,
  sidebar,
  conversation,
  input,
  sources,
}: AppShellProps) {
  const sidebarInline = useMediaQuery("(min-width: 768px)");
  const sourcesInline = useMediaQuery("(min-width: 1180px)");
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className={styles.shell}>
      {sidebarInline ? (
        <aside className={styles.sidebar} aria-label="Documents">
          {sidebar}
        </aside>
      ) : (
        <Sheet open={menuOpen} side="left" label="Documents" onClose={() => setMenuOpen(false)}>
          <div className={styles.drawer}>
            <button
              type="button"
              className={styles.drawerClose}
              aria-label="Close documents"
              onClick={() => setMenuOpen(false)}
            >
              <CloseIcon />
            </button>
            {sidebar}
          </div>
        </Sheet>
      )}

      <main className={styles.main}>
        <header className={styles.header}>
          {!sidebarInline && (
            <button type="button" className={styles.iconButton} aria-label="Open documents" onClick={() => setMenuOpen(true)}>
              <MenuIcon />
            </button>
          )}
          <p className={styles.context}>
            {documentName && (
              <>
                <span className={styles.contextLabel}>Asking</span>{" "}
                <span className={styles.contextName}>{documentName}</span>
              </>
            )}
          </p>
          <div className={styles.actions}>
            {hasTurns && (
              <button type="button" className={styles.textButton} onClick={onReset}>
                Clear
              </button>
            )}
            {!sourcesInline && sourceCount > 0 && (
              <button type="button" className={styles.textButton} onClick={() => onSourcesOpenChange(true)}>
                <PassagesIcon />
                {sourceCount} passages
              </button>
            )}
          </div>
        </header>
        <div className={styles.scroll}>{conversation}</div>
        {input}
      </main>

      {sourcesInline ? (
        <aside className={styles.sources} aria-label="Retrieved context">
          {sources}
        </aside>
      ) : (
        <Sheet open={sourcesOpen} side="right" label="Retrieved context" onClose={() => onSourcesOpenChange(false)}>
          {sources}
        </Sheet>
      )}
    </div>
  );
}
