"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CorpusStatus } from "@/components/CorpusStatus";
import { ResultRow } from "@/components/ResultRow";
import { ThreadDrawer } from "@/components/ThreadDrawer";
import { ThreadDrawerProvider, useThreadDrawer } from "@/components/ThreadDrawerContext";
import type { QueryThread, SearchThread } from "@/lib/backend";

const PYTHON_UI_URL = process.env.NEXT_PUBLIC_PYTHON_UI_URL ?? "http://127.0.0.1:8080";

// Adapt an inbox-shaped QueryThread into the SearchThread shape that
// ResultRow expects. Inbox rows have no per-message scores or match
// metadata, so we synthesise one "match" from the latest snippet and
// zero out everything else. This keeps the visual rendering identical
// to the search view for free.
const toSearchThread = (t: QueryThread): SearchThread => {
  // Prefer the actual sender of the latest message (backend populates
  // this). Fall back to participants[0] only for pre-upgrade responses.
  const topFrom = t.latest_from_addr ?? t.participants[0] ?? "?";
  return {
    thread_id: t.thread_id,
    score: 0,
    similarity: 0,
    subject: t.subject,
    participants: t.participants,
    message_count: t.message_count,
    date_first: t.date_first,
    date_last: t.date_last,
    user_replied: false,
    matches: [
      {
        message_id: t.latest_message_id ?? t.thread_id,
        score: 0,
        from_addr: topFrom,
        date: t.date_last,
        snippet: t.snippet,
        match_type: "inbox",
        attachment_filename: null,
        summary: t.summary,
        summary_model: t.summary_model ?? null,
        summary_created_at: t.summary_created_at ?? null,
      },
    ],
  };
};

const PAGE_SIZE = 50;

const InboxList = ({ threads, loading }: { threads: QueryThread[]; loading: boolean }) => {
  const { setOpenThreadId } = useThreadDrawer();
  if (loading && threads.length === 0) {
    return <div className="px-6 py-12 text-center text-sm text-muted-foreground">Loading…</div>;
  }
  if (!loading && threads.length === 0) {
    return (
      <div className="px-6 py-24 text-center text-sm text-muted-foreground">
        No priority messages. Gmail flags nothing in your inbox as IMPORTANT yet.
      </div>
    );
  }
  return (
    <div>
      {threads.map((t) => (
        <ResultRow key={t.thread_id} thread={toSearchThread(t)} onOpen={setOpenThreadId} />
      ))}
    </div>
  );
};

const DrawerHost = () => {
  const { openThreadId, setOpenThreadId } = useThreadDrawer();
  return (
    <ThreadDrawer threadId={openThreadId} onClose={() => setOpenThreadId(null)} pythonBaseUrl={PYTHON_UI_URL} />
  );
};

const InboxInner = () => {
  const [threads, setThreads] = useState<QueryThread[]>([]);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [reachedEnd, setReachedEnd] = useState(false);

  // "Time to glass": URL entered → all visible rows painted.
  //
  // Start: prefer the browser's Navigation Timing `startTime` when it's
  // recent (i.e. a full page load or reload is in progress). That's
  // the true "user hit this URL" moment. When the nav entry is stale
  // (SPA route change from within an already-loaded tab), fall back
  // to the earliest component-render clock we can reach — a ref
  // assigned during the first render pass, before any effect fires.
  //
  // End: after the threads state commits, wait two RAFs so the frame
  // the browser composites to screen has hit the compositor. First
  // RAF fires *before* the next paint; the second fires *after*.
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const mountStartRef = useRef<number | null>(null);
  if (mountStartRef.current === null) {
    const nav = (typeof performance !== "undefined"
      ? (performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined)
      : undefined);
    const now = performance.now();
    // If the nav entry looks recent (its responseEnd is within 2s of
    // now), the tab is mid-navigation — use its startTime. Otherwise
    // the nav is from the original page load minutes ago; fall back to
    // the render anchor so we're not showing the age of the whole
    // session.
    if (nav && now - nav.responseEnd < 2000) {
      mountStartRef.current = nav.startTime; // always 0 on reload/hard-load
    } else {
      mountStartRef.current = now;
    }
  }

  const loadMore = useCallback(
    async (nextOffset: number) => {
      setLoading(true);
      try {
        const res = await fetch(`/api/inbox?limit=${PAGE_SIZE}&offset=${nextOffset}`, { cache: "no-store" });
        if (!res.ok) throw new Error(`inbox ${res.status}`);
        const data = (await res.json()) as { results: QueryThread[] };
        const rows = data.results ?? [];
        setThreads((prev) => (nextOffset === 0 ? rows : [...prev, ...rows]));
        setReachedEnd(rows.length < PAGE_SIZE);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    loadMore(0);
  }, [loadMore]);

  // Stop the clock the first time the threads state commits with real
  // rows. Guard with a ref so Load More / subsequent re-renders don't
  // overwrite the first-paint number.
  useEffect(() => {
    if (mountStartRef.current === null) return;
    if (threads.length === 0) return;
    const start = mountStartRef.current;
    mountStartRef.current = null; // consume — this metric is per mount
    let raf2: number | null = null;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        setLatencyMs(performance.now() - start);
      });
    });
    return () => {
      cancelAnimationFrame(raf1);
      if (raf2 !== null) cancelAnimationFrame(raf2);
    };
  }, [threads]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b px-4 py-1.5">
        <CorpusStatus latencyMs={latencyMs} />
      </div>
      <header className="flex items-center justify-between border-b px-6 py-3">
        <div>
          <h1 className="text-base font-semibold">Priority inbox</h1>
          <p className="text-xs text-muted-foreground">
            Threads flagged IMPORTANT and still in the inbox, newest first.
          </p>
        </div>
        <div className="text-xs text-muted-foreground">{threads.length} threads</div>
      </header>
      <div className="flex-1 overflow-y-auto">
        <InboxList threads={threads} loading={loading} />
        {!reachedEnd && threads.length > 0 && (
          <div className="px-6 py-4 text-center">
            <button
              type="button"
              onClick={() => {
                const next = offset + PAGE_SIZE;
                setOffset(next);
                loadMore(next);
              }}
              disabled={loading}
              className="rounded-md border px-3 py-1 text-xs text-muted-foreground hover:bg-accent disabled:opacity-50"
            >
              {loading ? "Loading…" : "Load more"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export const InboxView = () => (
  <ThreadDrawerProvider>
    <InboxInner />
    <DrawerHost />
  </ThreadDrawerProvider>
);
