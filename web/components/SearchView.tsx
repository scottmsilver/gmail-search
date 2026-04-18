"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { CorpusStatus } from "@/components/CorpusStatus";
import { FacetSidebar } from "@/components/FacetSidebar";
import { ResultRow } from "@/components/ResultRow";
import { ThreadDrawer } from "@/components/ThreadDrawer";
import { ThreadDrawerProvider, useThreadDrawer } from "@/components/ThreadDrawerContext";
import type { SearchFacet, SearchResponse, SearchThread } from "@/lib/backend";

const PYTHON_UI_URL = process.env.NEXT_PUBLIC_PYTHON_UI_URL ?? "http://127.0.0.1:8080";

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
    return (
      <div className="px-6 py-12 text-center text-sm" style={{ color: "var(--fg-tertiary)" }}>
        Searching…
      </div>
    );
  }
  if (!query.trim()) {
    return (
      <div className="px-6 py-24 text-center" style={{ color: "var(--fg-tertiary)" }}>
        <div className="text-sm">Search your archive — try a person, topic, or phrase.</div>
      </div>
    );
  }
  if (results.length === 0) {
    return (
      <div className="px-6 py-12 text-center text-sm" style={{ color: "var(--fg-tertiary)" }}>
        No results for <span style={{ color: "var(--fg-secondary)" }}>“{query}”</span>.
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
  // Active topic is URL-driven: ?topic=root.0.1.0.1
  const activeTopic = urlTopic;
  const inputRef = useRef<HTMLInputElement>(null);
  const lastQueryRef = useRef<string>("");

  const runSearch = useCallback(
    async (q: string, s: Sort) => {
      const trimmed = q.trim();
      if (!trimmed) {
        setResults([]);
        setFacets([]);
        lastQueryRef.current = "";
        return;
      }
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
    <div
      className="flex flex-col h-full w-full"
      style={{ background: "var(--bg-primary)" }}
    >
      <div
        className="px-6 pt-6 pb-3 max-w-4xl w-full mx-auto"
        style={{ background: "var(--bg-primary)" }}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(query, sort);
          }}
          className="flex items-center gap-2 rounded-2xl border px-4 py-2.5 transition-colors focus-within:shadow-sm"
          style={{
            borderColor: "var(--border-subtle)",
            background: "var(--bg-secondary)",
          }}
        >
          <span style={{ color: "var(--fg-tertiary)" }}>
            <SearchIcon />
          </span>
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your archive…"
            autoFocus
            className="flex-1 bg-transparent focus:outline-none text-sm placeholder:opacity-60"
            style={{ color: "var(--fg-primary)" }}
          />
          <div
            className="flex items-center gap-0.5 rounded-full p-0.5 text-[11px]"
            style={{ background: "var(--bg-primary)" }}
          >
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
                    ? "px-2.5 py-0.5 rounded-full font-medium"
                    : "px-2.5 py-0.5 rounded-full opacity-60 hover:opacity-100"
                }
                style={
                  s === sort
                    ? { background: "var(--bg-secondary)", color: "var(--fg-primary)" }
                    : { color: "var(--fg-secondary)" }
                }
              >
                {s}
              </button>
            ))}
          </div>
        </form>
        <div className="mt-2">
          <CorpusStatus />
        </div>
      </div>
      <div className="flex-1 min-h-0 flex">
        <FacetSidebar
          facets={facets}
          totalCount={results.length}
          activeTopic={activeTopic}
          onSelectTopic={setActiveTopic}
        />
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto">
            <ResultsList results={visibleResults} loading={loading} query={lastQueryRef.current} />
          </div>
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
