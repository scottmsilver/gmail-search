"use client";

import { useEffect, useState } from "react";

import type { CorpusStatus as CorpusStatusType } from "@/lib/backend";
import { formatCalendarDate, formatSmartDate } from "@/lib/datetime";

// Mirrors the corpus status line shown in the search UI:
//   "35,861 messages · Apr 1, 2024 → 5m ago"
//
// Optional `latencyMs` appends a "time-to-glass" pill showing how long
// the last search took from fire→paint. Only SearchView passes this;
// Thread leaves it undefined (no search happening there).
type Props = { latencyMs?: number | null };

export const CorpusStatus = ({ latencyMs }: Props = {}) => {
  const [status, setStatus] = useState<CorpusStatusType | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as CorpusStatusType;
        if (!cancelled) setStatus(data);
      } catch {
        // ignore — transient network failure, retry on next tick
      }
    };
    void load();
    const id = setInterval(load, 60_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (!status) return null;

  const oldest = status.date_oldest ? formatCalendarDate(status.date_oldest) : "?";
  const latest = status.date_newest ? formatSmartDate(status.date_newest) : "?";

  // Format tiny dollar figures without trailing scientific junk.
  // "0.0025" → "$0.003"; "0.000002" → "<$0.01".
  const fmtCost = (usd: number): string => {
    if (usd <= 0) return "$0";
    if (usd < 0.01) return "<$0.01";
    if (usd < 1) return `$${usd.toFixed(2)}`;
    return `$${usd.toFixed(0)}`;
  };

  const corpusCost = fmtCost(status.total_cost_usd);
  const queryEmbeds = status.query_embeds ?? 0;
  const queryCostRaw = status.query_embed_cost_usd ?? 0;
  const queryCost = fmtCost(queryCostRaw);

  // "time to glass": last search's wall-clock from submit → paint.
  // < 1s shown in ms; ≥ 1s shown in seconds with one decimal.
  const fmtLatency = (ms: number): string => (ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`);

  return (
    <div
      className="truncate px-2 text-[10px] leading-tight text-muted-foreground"
      title={
        `Corpus spend: $${status.total_cost_usd.toFixed(4)}\n` +
        `Search embeds: ${queryEmbeds} distinct queries (cached; same text is free to re-search)\n` +
        `Search spend: $${queryCostRaw.toFixed(6)}` +
        (typeof latencyMs === "number" ? `\nLast search latency: ${latencyMs.toFixed(0)}ms` : "")
      }
    >
      {status.messages.toLocaleString()} · {oldest} → {latest} · corpus {corpusCost} · search {queryCost}
      {queryEmbeds > 0 && <> ({queryEmbeds.toLocaleString()})</>}
      {typeof latencyMs === "number" && <> · glass {fmtLatency(latencyMs)}</>}
    </div>
  );
};
