"use client";

import { useEffect, useState } from "react";

import type { CorpusStatus as CorpusStatusType } from "@/lib/backend";

const RTF = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

const formatRelative = (iso: string): string => {
  const dt = new Date(iso);
  if (isNaN(dt.getTime())) return "?";
  const diffSec = Math.floor((Date.now() - dt.getTime()) / 1000);
  if (diffSec < 60) return RTF.format(-diffSec, "second");
  if (diffSec < 3600) return RTF.format(-Math.floor(diffSec / 60), "minute");
  if (diffSec < 86400) return RTF.format(-Math.floor(diffSec / 3600), "hour");
  if (diffSec < 604800) return RTF.format(-Math.floor(diffSec / 86400), "day");
  return dt.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
};

const formatDate = (iso: string): string => {
  const dt = new Date(iso);
  if (isNaN(dt.getTime())) return "?";
  return dt.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
};

// Mirrors the corpus status line shown in the search UI:
//   "35,861 messages · Apr 1 2024 to 5 minutes ago"
// plus a sync indicator when the watch daemon is mid-cycle.
export const CorpusStatus = () => {
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

  const oldest = status.date_oldest ? formatDate(status.date_oldest) : "?";
  const latest = status.date_newest ? formatRelative(status.date_newest) : "?";

  return (
    <div className="text-[10px] text-neutral-400 px-2 leading-tight truncate">
      {status.messages.toLocaleString()} · {oldest} → {latest}
    </div>
  );
};
