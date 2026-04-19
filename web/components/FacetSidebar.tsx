"use client";

import { cn } from "@/lib/utils";
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
    <aside className="hidden w-56 shrink-0 overflow-y-auto border-r bg-background md:block">
      <div className="px-4 pb-2 pt-4 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        Topics
      </div>
      <FacetButton active={activeTopic === null} onClick={() => onSelectTopic(null)} label="All results" count={totalCount} />
      <div className="mt-1">
        {facets.map((f) => (
          <FacetButton
            key={f.topic_id}
            active={activeTopic === f.topic_id}
            onClick={() => onSelectTopic(activeTopic === f.topic_id ? null : f.topic_id)}
            label={f.label}
            count={f.count}
            title={f.topic_id}
          />
        ))}
      </div>
    </aside>
  );
};

const FacetButton = ({
  active,
  onClick,
  label,
  count,
  title,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count: number;
  title?: string;
}) => (
  <button
    type="button"
    onClick={onClick}
    title={title}
    className={cn(
      "flex w-full items-center justify-between px-4 py-1.5 text-left text-xs transition-colors",
      active
        ? "bg-secondary font-semibold text-foreground"
        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
    )}
  >
    <span className="truncate">{label}</span>
    <span className="ml-2 shrink-0 text-muted-foreground">{count}</span>
  </button>
);
