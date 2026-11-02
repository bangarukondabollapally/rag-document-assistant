import type { DocumentSummary, IndexStatus, PipelineConfig, Workspace } from "../api/types";
import styles from "./Sidebar.module.css";

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

function DocumentItem({ document, index }: { document: DocumentSummary; index: IndexStatus }) {
  const statusLabel = { loading: "Loading index", ready: "Indexed", not_indexed: "Not indexed", failed: "Failed" }[
    document.status
  ];
  return (
    <li className={styles.document}>
      <div className={styles.documentName}>{document.name}</div>
      {document.title && <div className={styles.documentTitle}>{document.title}</div>}
      <div className={`${styles.status} ${styles[document.status]}`}>
        <span className={styles.dot} aria-hidden="true" />
        {statusLabel}
      </div>
      {document.status === "ready" && document.pages_indexed !== null && (
        <div className={styles.meta}>
          {document.pages_indexed} of {document.pages_total} pages · {document.chunks?.toLocaleString()} chunks
        </div>
      )}
      {document.status === "loading" && (
        <p className={styles.note}>
          {index.stage === "checking" ? "Checking the saved index." : "Loading the saved index."}{" "}
          {formatDuration(index.elapsed_seconds)} elapsed.
        </p>
      )}
      {document.status === "not_indexed" && index.reason && <p className={styles.note}>{index.reason}</p>}
      {document.status === "failed" && index.error && <p className={styles.errorText}>{index.error}</p>}
    </li>
  );
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
}

export function Sidebar({ workspace, unreachable }: SidebarProps) {
  return (
    <div className={styles.sidebar}>
      <h1 className={styles.product}>RAG Document Assistant</h1>

      <section className={styles.section}>
        <h2 className={styles.label}>Document</h2>
        {unreachable && (
          <p className={styles.errorText}>{workspace ? "API unreachable. Showing the last known state." : "API unreachable."}</p>
        )}
        {workspace ? (
          <ul className={styles.documents}>
            {workspace.documents.map((document) => (
              <DocumentItem key={document.id} document={document} index={workspace.index} />
            ))}
          </ul>
        ) : (
          !unreachable && <p className={styles.note}>Connecting…</p>
        )}
        {workspace && !workspace.capabilities.upload && (
          <p className={styles.note}>This version reads one PDF, set on the server. Upload isn't built yet.</p>
        )}
      </section>

      {workspace && <PipelineSummary pipeline={workspace.pipeline} />}
    </div>
  );
}
