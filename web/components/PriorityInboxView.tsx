"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CorpusStatus } from "@/components/CorpusStatus";
import { ResultRow } from "@/components/ResultRow";
import { ThreadDrawer } from "@/components/ThreadDrawer";
import { ThreadDrawerProvider, useThreadDrawer } from "@/components/ThreadDrawerContext";
import type { PriorityInboxSection, QueryThread, SearchThread } from "@/lib/backend";

const PYTHON_UI_URL = process.env.NEXT_PUBLIC_PYTHON_UI_URL ?? "http://127.0.0.1:8080";

// Reuse the same adapter pattern as InboxView — inbox rows have no
// per-message score or match metadata, so we synthesise a single "match"
// from the latest snippet. Keeps rendering identical to /search and
// /inbox for free.
const toSearchThread = (t: QueryThread): SearchThread => {
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

const PAGE_SIZE = 25;

const SectionBlock = ({ section }: { section: PriorityInboxSection }) => {
  const { setOpenThreadId } = useThreadDrawer();
  return (
    <section>
      {/* Sticky header so the label stays on-screen while scrolling the
          section's rows. Gmail renders the section title persistent-top
          within each block. */}
      <header className="sticky top-0 z-10 flex items-center justify-between gap-2 border-b bg-background/95 px-3 py-1.5 backdrop-blur">
        <h2 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {section.title}
        </h2>
        <span className="shrink-0 whitespace-nowrap text-[10px] text-muted-foreground/80">
          {section.threads.length === 0
            ? "(none)"
            : `${section.threads.length} ${section.threads.length === 1 ? "thread" : "threads"}`}
        </span>
      </header>
      {section.threads.length > 0 ? (
        <div>
          {section.threads.map((t) => (
            <ResultRow key={t.thread_id} thread={toSearchThread(t)} onOpen={setOpenThreadId} />
          ))}
        </div>
      ) : null}
    </section>
  );
};

const DrawerHost = () => {
  const { openThreadId, setOpenThreadId } = useThreadDrawer();
  return (
    <ThreadDrawer threadId={openThreadId} onClose={() => setOpenThreadId(null)} pythonBaseUrl={PYTHON_UI_URL} />
  );
};

const PriorityInboxInner = () => {
  const [sections, setSections] = useState<PriorityInboxSection[]>([]);
  const [loading, setLoading] = useState(true);

  // Same "time to glass" instrumentation as InboxView: prefer the
  // browser's Navigation Timing `startTime` when recent, else the
  // earliest render clock. Stop after threads commit + two RAFs.
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const mountStartRef = useRef<number | null>(null);
  if (mountStartRef.current === null) {
    const nav = (typeof performance !== "undefined"
      ? (performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined)
      : undefined);
    const now = performance.now();
    if (nav && now - nav.responseEnd < 2000) {
      mountStartRef.current = nav.startTime;
    } else {
      mountStartRef.current = now;
    }
  }

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/priority-inbox?limit=${PAGE_SIZE}&offset=0`, { cache: "no-store" });
      if (!res.ok) throw new Error(`priority-inbox ${res.status}`);
      const data = (await res.json()) as { sections: PriorityInboxSection[] };
      setSections(data.sections ?? []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (mountStartRef.current === null) return;
    if (sections.length === 0) return;
    const start = mountStartRef.current;
    mountStartRef.current = null;
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
  }, [sections]);

  const totalThreads = sections.reduce((sum, s) => sum + s.threads.length, 0);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex-1 overflow-y-auto">
        {loading && sections.length === 0 ? (
          <div className="px-6 py-12 text-center text-sm text-muted-foreground">Loading…</div>
        ) : (
          sections.map((s) => <SectionBlock key={s.key} section={s} />)
        )}
      </div>
      {/* Same bottom status strip as InboxView. */}
      <div className="flex shrink-0 items-center justify-between gap-3 border-t px-2 py-1.5">
        <div className="min-w-0 flex-1">
          <CorpusStatus latencyMs={latencyMs} />
        </div>
        <span className="shrink-0 whitespace-nowrap text-[10px] text-muted-foreground/80">
          {totalThreads} {totalThreads === 1 ? "thread" : "threads"}
        </span>
      </div>
    </div>
  );
};

export const PriorityInboxView = () => (
  <ThreadDrawerProvider>
    <PriorityInboxInner />
    <DrawerHost />
  </ThreadDrawerProvider>
);
