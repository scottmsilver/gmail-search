"use client";

import type { SearchFacet } from "@/lib/backend";

type Props = {
  facets: SearchFacet[];
  totalCount: number;
  activeTopic: string | null;
  onSelectTopic: (topicId: string | null) => void;
};

export const FacetSidebar = ({ facets, totalCount, activeTopic, onSelectTopic }: Props) => {
  if (facets.length === 0 && totalCount === 0) return null;

  return (
    <aside
      className="hidden md:block w-56 shrink-0 border-r overflow-y-auto"
      style={{ borderColor: "var(--border-subtle)", background: "var(--bg-primary)" }}
    >
      <div
        className="px-4 pt-4 pb-2 text-[10px] uppercase tracking-wide font-semibold"
        style={{ color: "var(--fg-tertiary)" }}
      >
        Topics
      </div>
      <button
        type="button"
        onClick={() => onSelectTopic(null)}
        className="w-full text-left px-4 py-1.5 text-xs flex items-center justify-between transition-colors theme-hover"
        style={{
          color: activeTopic === null ? "var(--fg-primary)" : "var(--fg-secondary)",
          fontWeight: activeTopic === null ? 600 : 400,
          background: activeTopic === null ? "var(--bg-secondary)" : undefined,
        }}
      >
        <span>All results</span>
        <span style={{ color: "var(--fg-tertiary)" }}>{totalCount}</span>
      </button>
      <div className="mt-1">
        {facets.map((f) => {
          const active = activeTopic === f.topic_id;
          return (
            <button
              key={f.topic_id}
              type="button"
              onClick={() => onSelectTopic(active ? null : f.topic_id)}
              className="w-full text-left px-4 py-1.5 text-xs flex items-center justify-between transition-colors theme-hover"
              style={{
                color: active ? "var(--fg-primary)" : "var(--fg-secondary)",
                fontWeight: active ? 600 : 400,
                background: active ? "var(--bg-secondary)" : undefined,
              }}
              title={f.topic_id}
            >
              <span className="truncate">{f.label}</span>
              <span className="ml-2 shrink-0" style={{ color: "var(--fg-tertiary)" }}>
                {f.count}
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
};
