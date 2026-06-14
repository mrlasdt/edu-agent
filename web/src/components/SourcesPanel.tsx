import type { Citation } from "../types";

interface Props {
  citations: Citation[];
  isOpen: boolean;
  onClose: () => void;
}

const TIER_LABELS: Record<string, string> = {
  global: "Official ETS material",
  school: "Course notes",
  candidate: "Your notes",
};

export function SourcesPanel({ citations, isOpen, onClose }: Props) {
  if (!isOpen) return null;

  return (
    <aside className="sources-panel" aria-label="Sources">
      <div className="sources-panel__header">
        <h3>Sources</h3>
        <button onClick={onClose} aria-label="Close sources panel">
          Close ✕
        </button>
      </div>
      <ul className="sources-list">
        {citations.map((c) => (
          <li key={c.index} className="source-item">
            <span className="source-index">[{c.index}]</span>
            <div className="source-meta">
              <span className="source-uri">{c.source_uri}</span>
              {" — "}
              <span className="source-section">{c.page_or_section}</span>
              {" — "}
              <span className="source-tier">{TIER_LABELS[c.tier] ?? c.tier}</span>
            </div>
            <p className="source-text">{c.text}</p>
          </li>
        ))}
      </ul>
    </aside>
  );
}
