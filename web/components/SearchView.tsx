"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { CorpusStatus } from "@/components/CorpusStatus";
import { FacetSidebar } from "@/components/FacetSidebar";
import { ResultRow } from "@/components/ResultRow";
import { ThreadDrawer } from "@/components/ThreadDrawer";
import { ThreadDrawerProvider, useThreadDrawer } from "@/components/ThreadDrawerContext";
import type { SearchFacet, SearchResponse, SearchThread } from "@/lib/backend";

const PYTHON_UI_URL = process.env.NEXT_PUBLIC_PYTHON_UI_URL ?? "";

type Sort = "relevance" | "recent";

const SearchIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
    <circle cx="11" cy="11" r="7" />
    <path strokeLinecap="round" d="M20 20l-3.5-3.5" />
  </svg>
);

const ResultsList = ({
  results,
  loading,
  query,
}: {
  results: SearchThread[];
  loading: boolean;
  query: string;
}) => {
  const { setOpenThreadId } = useThreadDrawer();
  if (loading) {
    return <div className="px-6 py-12 text-center text-sm text-muted-foreground">Searching…</div>;
  }
  if (!query.trim()) {
    return (
      <div className="px-6 py-24 text-center text-sm text-muted-foreground">
        Search your archive — try a person, topic, or phrase.
      </div>
    );
  }
  if (results.length === 0) {
    return (
      <div className="px-6 py-12 text-center text-sm text-muted-foreground">
        No results for <span className="text-foreground">“{query}”</span>.
      </div>
    );
  }
  return (
    <div>
      {results.map((t) => (
        <ResultRow key={t.thread_id} thread={t} onOpen={setOpenThreadId} />
      ))}
    </div>
  );
};

const DrawerHost = () => {
  const { openThreadId, setOpenThreadId } = useThreadDrawer();
  return (
    <ThreadDrawer
      threadId={openThreadId}
      onClose={() => setOpenThreadId(null)}
      pythonBaseUrl={PYTHON_UI_URL}
    />
  );
};

const SearchInner = () => {
  const router = useRouter();
  const params = useSearchParams();
  const initialQuery = params.get("q") ?? "";
  const initialSort = (params.get("sort") as Sort | null) ?? "relevance";
  const urlTopic = params.get("topic");

  const [query, setQuery] = useState(initialQuery);
  const [sort, setSort] = useState<Sort>(initialSort);
  const [results, setResults] = useState<SearchThread[]>([]);
  const [facets, setFacets] = useState<SearchFacet[]>([]);
  const [loading, setLoading] = useState(false);
  const [showTopics, setShowTopics] = useState(false);
  // Active topic is URL-driven: ?topic=root.0.1.0.1
  const activeTopic = urlTopic;
  const inputRef = useRef<HTMLInputElement>(null);
  const lastQueryRef = useRef<string>("");
  // "Time to glass": ms from search-submit to the frame where the
  // results become visible on screen. Measured by snapshotting
  // performance.now() before fetch and reading it back in a RAF
  // scheduled after the results state commits (see useEffect on
  // `results` below). Displayed next to the cost pill in CorpusStatus.
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const searchStartRef = useRef<number | null>(null);

  const runSearch = useCallback(
    async (q: string, s: Sort) => {
      const trimmed = q.trim();
      if (!trimmed) {
        setResults([]);
        setFacets([]);
        lastQueryRef.current = "";
        searchStartRef.current = null;
        return;
      }
      // Anchor the stopwatch at the moment of submit — fetch + parse
      // + React render + browser paint all fold into this window.
      searchStartRef.current = performance.now();
      setLoading(true);
      try {
        const url = new URL("/api/search", window.location.origin);
        url.searchParams.set("q", trimmed);
        url.searchParams.set("k", "30");
        url.searchParams.set("sort", s);
        const res = await fetch(url.toString(), { cache: "no-store" });
        if (!res.ok) {
          setResults([]);
          setFacets([]);
          return;
        }
        const data = (await res.json()) as SearchResponse;
        setResults(data.results ?? []);
        setFacets(data.facets ?? []);
        lastQueryRef.current = trimmed;
      } catch {
        setResults([]);
        setFacets([]);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // After each results state commit, schedule two RAFs and measure at
  // the second — by then the browser has laid out the rows and
  // painted the frame the user sees. Single RAF fires *before* the
  // paint; the second fires *after*, which is the correct "glass"
  // moment.
  useEffect(() => {
    if (searchStartRef.current === null) return;
    const start = searchStartRef.current;
    searchStartRef.current = null; // consume; next search will re-anchor
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
  }, [results]);

  // Push topic selection into the URL (preserves q/sort/thread). Read the
  // live URL at click time so a near-simultaneous setOpenThreadId doesn't
  // clobber us via a stale params snapshot.
  const setActiveTopic = useCallback(
    (topicId: string | null) => {
      const live = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
      if (topicId && /^[a-zA-Z0-9._-]{1,128}$/.test(topicId)) live.set("topic", topicId);
      else live.delete("topic");
      const qs = live.toString();
      router.replace(qs ? `/search?${qs}` : "/search", { scroll: false });
    },
    [router],
  );

  // Run search on mount if URL had a query param.
  useEffect(() => {
    if (initialQuery) void runSearch(initialQuery, initialSort);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Gmail-style search-as-you-type. Fires 250ms after the last
  // keystroke — short enough to feel live, long enough that each
  // letter of a word-in-progress doesn't trigger a backend hit.
  // Cancels on unmount / further typing so we never race stale
  // responses against the current query.
  useEffect(() => {
    const trimmed = query.trim();
    // Skip if the query hasn't changed since the last successful
    // search — prevents a redundant fetch right after `submit` runs.
    if (trimmed === lastQueryRef.current) return;
    const t = setTimeout(() => {
      submit(query, sort);
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, sort]);

  const submit = useCallback(
    (q: string, s: Sort) => {
      const live = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
      if (q.trim()) live.set("q", q.trim());
      else live.delete("q");
      if (s !== "relevance") live.set("sort", s);
      else live.delete("sort");
      // New search resets the topic filter — old facets won't apply.
      live.delete("topic");
      const qs = live.toString();
      router.replace(qs ? `/search?${qs}` : "/search", { scroll: false });
      void runSearch(q, s);
    },
    [router, runSearch],
  );

  const visibleResults = activeTopic
    ? results.filter((r) => (r.topic_ids ?? []).includes(activeTopic))
    : results;

  return (
    <div className="flex h-full w-full flex-col bg-background">
      <div className="mx-auto w-full max-w-4xl px-6 pb-3 pt-6">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(query, sort);
          }}
          className="flex items-center gap-2 rounded-2xl border bg-card px-4 py-2.5 transition-colors focus-within:shadow-sm"
        >
          <span className="text-muted-foreground">
            <SearchIcon />
          </span>
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your archive…"
            autoFocus
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
          <div className="flex items-center gap-0.5 rounded-full bg-muted p-0.5 text-[11px]">
            {(["relevance", "recent"] as Sort[]).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => {
                  setSort(s);
                  if (lastQueryRef.current) submit(query, s);
                }}
                className={
                  s === sort
                    ? "rounded-full bg-background px-2.5 py-0.5 font-medium text-foreground shadow-sm"
                    : "rounded-full px-2.5 py-0.5 text-muted-foreground hover:text-foreground"
                }
              >
                {s}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setShowTopics((v) => !v)}
            aria-label="Toggle topics"
            title={showTopics ? "Hide topics" : "Show topics"}
            className={
              showTopics
                ? "rounded-full bg-secondary px-2.5 py-1 text-[11px] font-medium text-foreground"
                : "rounded-full px-2.5 py-1 text-[11px] text-muted-foreground hover:text-foreground"
            }
          >
            topics
          </button>
        </form>
        <div className="mt-2">
          <CorpusStatus latencyMs={latencyMs} />
        </div>
      </div>
      <div className="flex min-h-0 flex-1">
        {showTopics && (
          <FacetSidebar
            facets={facets}
            totalCount={results.length}
            activeTopic={activeTopic}
            onSelectTopic={setActiveTopic}
          />
        )}
        <div className="flex-1 overflow-y-auto">
          {/* Results span the full content width (minus the topics
              sidebar). Wide monitors get richer summaries without
              wrapping into narrow columns. */}
          <ResultsList results={visibleResults} loading={loading} query={lastQueryRef.current} />
        </div>
      </div>
    </div>
  );
};

export const SearchView = () => (
  <ThreadDrawerProvider>
    <SearchInner />
    <DrawerHost />
  </ThreadDrawerProvider>
);
