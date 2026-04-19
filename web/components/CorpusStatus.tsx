"use client";

import { useEffect, useState } from "react";

import type { CorpusStatus as CorpusStatusType } from "@/lib/backend";
import { formatCalendarDate, formatSmartDate } from "@/lib/datetime";

// Mirrors the corpus status line shown in the search UI:
//   "35,861 messages · Apr 1, 2024 → 5m ago"
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

  const oldest = status.date_oldest ? formatCalendarDate(status.date_oldest) : "?";
  const latest = status.date_newest ? formatSmartDate(status.date_newest) : "?";

  return (
    <div className="truncate px-2 text-[10px] leading-tight text-muted-foreground">
      {status.messages.toLocaleString()} · {oldest} → {latest}
    </div>
  );
};
