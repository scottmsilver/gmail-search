"use client";

import { useCallback, useRef, useState } from "react";

import { runAgentAnalyze, type AgentEvent } from "@/lib/agentStream";

import { CitableMarkdown } from "./CitableMarkdown";
import { ThreadDrawerProvider } from "./ThreadDrawerContext";

// Deep-analysis UI (Phase 6c): minimal textarea + stream of stage
// events + final markdown. No chat-style multi-turn — each
// submission starts its own ephemeral session. The agent transcript
// appears progressively as the backend emits SSE frames.
//
// Event layout top-to-bottom (matches orchestration order):
//   session → plan → evidence → analysis (or skipped)
//   → draft → critique [→ revision → critique…]
//   → final
// The final markdown is rendered through CitableMarkdown so [ref:],
// [att:], and the new [art:] chips all work the same as in chat mode.

type StageBlock = {
  kind: string;
  agent?: string;
  payload?: unknown;
  seq?: number;
};

const friendlyLabels: Record<string, string> = {
  session: "Session started",
  plan: "Plan",
  evidence: "Retrieved",
  analysis: "Analyst",
  skipped: "Analyst skipped",
  draft: "Draft",
  critique: "Critique",
  revision: "Revision",
  final: "Final answer",
  error: "Error",
};

const StageCard = ({ block }: { block: StageBlock }) => {
  const label = friendlyLabels[block.kind] ?? block.kind;
  const payload = block.payload as Record<string, unknown> | undefined;
  const isFinal = block.kind === "final";
  const isError = block.kind === "error";
  const text = typeof payload?.text === "string" ? (payload.text as string) : null;

  // Final bubble gets markdown rendering with chip citations.
  if (isFinal && text) {
    return (
      <div className="rounded border border-emerald-200 bg-emerald-50 p-3">
        <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-emerald-800">Final answer</div>
        <CitableMarkdown text={text} hints={[]} />
      </div>
    );
  }
  return (
    <div
      className={`rounded border p-2 text-xs ${
        isError ? "border-red-200 bg-red-50 text-red-900" : "border-neutral-200 bg-neutral-50 text-neutral-700"
      }`}
    >
      <div className="mb-0.5 font-medium">
        {label}
        {block.agent && block.agent !== block.kind && (
          <span className="ml-1 text-neutral-400">· {block.agent}</span>
        )}
        {typeof block.seq === "number" && <span className="ml-1 text-neutral-400">#{block.seq}</span>}
      </div>
      <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">
        {safeStringify(payload)}
      </pre>
    </div>
  );
};

const safeStringify = (v: unknown): string => {
  if (v === undefined || v === null) return "";
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
};

const DeepAnalysisInner = () => {
  const [question, setQuestion] = useState("");
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<StageBlock[]>([]);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const submit = useCallback(async () => {
    if (!question.trim() || running) return;
    setRunning(true);
    setError(null);
    setEvents([]);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await runAgentAnalyze(
        { question },
        {
          signal: ctrl.signal,
          onEvent: (ev: AgentEvent) => {
            setEvents((prev) => [
              ...prev,
              {
                kind: ev.kind,
                agent: ev.agent,
                payload: ev.payload ?? ev.raw,
                seq: ev.seq,
              },
            ]);
          },
          onError: (e: unknown) => setError(e instanceof Error ? e.message : String(e)),
        },
      );
    } finally {
      setRunning(false);
    }
  }, [question, running]);

  const cancel = () => {
    abortRef.current?.abort();
  };

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-3 px-6 py-8">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">Deep Analysis</h1>
        <p className="text-sm text-muted-foreground">
          Opt-in multi-agent mode with a Python sandbox. Slower and more expensive than chat, but
          can run real analysis (aggregations, plots, clustering) across your archive.
        </p>
      </div>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="What do you want to know? e.g. 'plot my monthly spending by sender over the last year'"
        className="min-h-[80px] w-full resize-y rounded border border-neutral-200 bg-background p-2 text-sm"
        disabled={running}
      />

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={submit}
          disabled={running || !question.trim()}
          className="rounded bg-foreground px-3 py-1.5 text-xs font-medium text-background hover:opacity-90 disabled:opacity-40"
        >
          {running ? "Analyzing…" : "Analyze"}
        </button>
        {running && (
          <button
            type="button"
            onClick={cancel}
            className="rounded border border-neutral-200 px-3 py-1.5 text-xs text-neutral-600 hover:bg-neutral-50"
          >
            Cancel
          </button>
        )}
        {error && <span className="text-xs text-red-600">{error}</span>}
      </div>

      {events.length > 0 && (
        <div className="flex flex-col gap-2">
          {events.map((ev, i) => (
            <StageCard key={`${ev.seq ?? i}-${ev.kind}`} block={ev} />
          ))}
        </div>
      )}
    </div>
  );
};

export const DeepAnalysisView = () => (
  <ThreadDrawerProvider>
    <DeepAnalysisInner />
  </ThreadDrawerProvider>
);
