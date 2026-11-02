import { useState } from "react";
import type { DocumentSummary, IndexStatus, PipelineConfig, Workspace } from "../api/types";
import type { DocumentsState, PendingAction } from "../state/useDocuments";
import { UploadButton } from "./UploadButton";
import styles from "./Sidebar.module.css";

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

function statusText(document: DocumentSummary, pending: PendingAction | undefined): string {
  if (pending === "deleting") return "Deleting…";
  switch (document.status) {
    case "uploaded":
      return "Waiting to index";
    case "indexing":
      return "Indexing…";
    case "failed":
      return "Failed";
    case "ready":
      return `Ready · ${(document.pages_indexed ?? 0).toLocaleString()} pages · ${(document.chunks ?? 0).toLocaleString()} chunks`;
  }
}

interface DocumentItemProps {
  document: DocumentSummary;
  checked: boolean;
  pending: PendingAction | undefined;
  onToggle: () => void;
  onDelete: () => void;
  onReindex: () => void;
}

function DocumentItem({ document, checked, pending, onToggle, onDelete, onReindex }: DocumentItemProps) {
  const [confirming, setConfirming] = useState<"delete" | "reindex" | null>(null);
  const processing = document.status === "uploaded" || document.status === "indexing";
  const statusClass = processing || pending === "deleting" ? styles.processing : styles[document.status];

  return (
    <li className={styles.document}>
      <label className={styles.documentSelect}>
        <input type="checkbox" checked={checked} disabled={document.status !== "ready"} onChange={onToggle} />
        <span className={styles.documentName}>{document.name}</span>
      </label>
      <div className={styles.documentBody}>
        {document.title && <div className={styles.documentTitle}>{document.title}</div>}
        <div className={`${styles.status} ${statusClass}`}>
          <span className={styles.dot} aria-hidden="true" />
          {statusText(document, pending)}
        </div>
        {document.status === "failed" && document.error && <p className={styles.errorText}>{document.error}</p>}

        {!processing && !pending && confirming === null && (
          <div className={styles.actions}>
            {document.status === "failed" ? (
              <button type="button" className={styles.action} onClick={onReindex}>
                Retry indexing
              </button>
            ) : (
              <button type="button" className={styles.action} onClick={() => setConfirming("reindex")}>
                Reindex
              </button>
            )}
            <button type="button" className={styles.action} onClick={() => setConfirming("delete")}>
              Delete
            </button>
          </div>
        )}

        {!pending && confirming !== null && (
          <div className={styles.actions} role="group" aria-label={`Confirm for ${document.name}`}>
            <span className={styles.confirmText}>
              {confirming === "delete" ? "Delete this document and its index?" : "Rebuild this document's index?"}
            </span>
            <button
              type="button"
              className={`${styles.action} ${styles.confirm}`}
              onClick={() => {
                setConfirming(null);
                if (confirming === "delete") onDelete();
                else onReindex();
              }}
            >
              {confirming === "delete" ? "Yes, delete" : "Yes, reindex"}
            </button>
            <button type="button" className={styles.action} onClick={() => setConfirming(null)}>
              Cancel
            </button>
          </div>
        )}
      </div>
    </li>
  );
}

function SearchEngineStatus({ index }: { index: IndexStatus }) {
  if (index.state === "failed" && index.error) return <p className={styles.errorText}>{index.error}</p>;
  if (index.state !== "loading") return null;
  return <p className={styles.note}>Loading the search index. {formatDuration(index.elapsed_seconds)} elapsed.</p>;
}

function PipelineSummary({ pipeline }: { pipeline: PipelineConfig }) {
  const rows: [string, string][] = [
    ["Chunks", `${pipeline.chunk_size_chars} chars, ${pipeline.chunk_overlap_chars} overlap`],
    ["Embeddings", pipeline.embedding_model.split("/").pop() ?? pipeline.embedding_model],
    ["Index", pipeline.vector_store],
    ["Retrieval", `${pipeline.search_type}, ${pipeline.k} of ${pipeline.fetch_k}, λ ${pipeline.lambda}`],
    ["Model", pipeline.llm_model],
  ];
  return (
    <section className={styles.section}>
      <h2 className={styles.label}>Pipeline</h2>
      <dl className={styles.pipeline}>
        {rows.map(([term, value]) => (
          <div key={term} className={styles.pipelineRow}>
            <dt>{term}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

interface SidebarProps {
  workspace: Workspace | null;
  unreachable: boolean;
  documents: DocumentsState;
}

export function Sidebar({ workspace, unreachable, documents }: SidebarProps) {
  const isEmpty = workspace?.documents.length === 0 && documents.uploads.length === 0;
  return (
    <div className={styles.sidebar}>
      <h1 className={styles.product}>RAG Document Assistant</h1>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.label}>Documents</h2>
          {workspace && <UploadButton disabled={unreachable} onUpload={(file) => void documents.upload(file)} />}
        </div>
        {unreachable && (
          <p className={styles.errorText}>{workspace ? "API unreachable. Showing the last known state." : "API unreachable."}</p>
        )}
        {documents.error && (
          <p className={styles.errorText} role="alert">
            {documents.error}
          </p>
        )}
        {workspace ? (
          <>
            <SearchEngineStatus index={workspace.index} />
            {isEmpty ? (
              <div className={styles.emptyDocuments}>
                <p className={styles.emptyTitle}>No documents yet</p>
                <p className={styles.note}>Upload a PDF to get started</p>
              </div>
            ) : (
              <ul className={styles.documents}>
                {workspace.documents.map((document) => (
                  <DocumentItem
                    key={document.id}
                    document={document}
                    checked={documents.isChecked(document.id)}
                    pending={documents.pending[document.id]}
                    onToggle={() => documents.toggle(document.id)}
                    onDelete={() => void documents.remove(document.id)}
                    onReindex={() => void documents.reindex(document.id)}
                  />
                ))}
                {documents.uploads.map((name, position) => (
                  <li key={`${name}-${position}`} className={styles.document}>
                    <div className={styles.documentSelect}>
                      <input type="checkbox" disabled aria-label={`${name} isn't ready`} />
                      <span className={styles.documentName}>{name}</span>
                    </div>
                    <div className={styles.documentBody}>
                      <div className={`${styles.status} ${styles.processing}`}>
                        <span className={styles.dot} aria-hidden="true" />
                        Uploading…
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </>
        ) : (
          !unreachable && <p className={styles.note}>Connecting…</p>
        )}
      </section>

      {workspace && <PipelineSummary pipeline={workspace.pipeline} />}
    </div>
  );
}
