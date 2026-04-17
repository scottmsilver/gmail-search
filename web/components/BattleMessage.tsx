"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { extractThreadHints } from "@/lib/extractThreadHints";
import { linkifyRefs, REF_PREFIX } from "@/lib/linkifyRefs";
import { variantLabel, type BattleVariant } from "@/lib/battleVariants";

import { CitationChip } from "./CitationChip";
import { useThreadDrawer } from "./ThreadDrawerContext";

type BattleData = {
  request_id: string;
  question: string;
  variant_a: BattleVariant;
  variant_b: BattleVariant;
  answer_a: string;
  answer_b: string;
  tools_a: Array<{ name: string; args: unknown; output: unknown }>;
  tools_b: Array<{ name: string; args: unknown; output: unknown }>;
  running_a?: boolean;
  running_b?: boolean;
};

const Spinner = () => (
  <svg className="w-4 h-4 animate-spin text-neutral-400" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth={3} opacity={0.25} />
    <path d="M22 12a10 10 0 0 1-10 10" stroke="currentColor" strokeWidth={3} strokeLinecap="round" />
  </svg>
);

// Walk tool results and strip huge base64 blobs so the inspector JSON
// stays readable. Same logic as the main ToolCallUI inspector.
const stripBinary = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(stripBinary);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (k === "base64" && typeof v === "string") {
        out[k] = `<${v.length} chars base64 stripped>`;
      } else {
        out[k] = stripBinary(v);
      }
    }
    return out;
  }
  return value;
};

const formatJson = (v: unknown): string => {
  try {
    return JSON.stringify(stripBinary(v), null, 2);
  } catch {
    return String(v);
  }
};

type BattleTool = { name: string; args: unknown; output: unknown };

const InspectorDrawer = ({
  open,
  onClose,
  label,
  tools,
}: {
  open: boolean;
  onClose: () => void;
  label: string;
  tools: BattleTool[];
}) => {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <>
      <div
        className={`fixed inset-0 bg-black/20 transition-opacity z-40 ${
          open ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
        onClick={onClose}
      />
      <aside
        className={`fixed top-0 right-0 h-full w-full sm:w-[560px] bg-white shadow-2xl z-50 transition-transform duration-200 flex flex-col ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
        aria-hidden={!open}
      >
        <header className="px-5 py-4 border-b flex items-start gap-3">
          <div className="flex-1 min-w-0">
            <h2 className="font-semibold text-sm text-neutral-900 truncate">Inspector — {label}</h2>
            <p className="text-xs text-neutral-500 mt-0.5">
              {tools.length} tool call{tools.length === 1 ? "" : "s"}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-neutral-500 hover:bg-neutral-100"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </header>
        <div className="flex-1 overflow-y-auto px-5 py-3 space-y-4">
          {tools.length === 0 && (
            <div className="text-xs text-neutral-400">No tool calls for this variant.</div>
          )}
          {tools.map((t, i) => (
            <div key={i} className="rounded border border-neutral-200 overflow-hidden">
              <div className="bg-neutral-50 px-3 py-2 text-xs font-mono text-neutral-700 border-b border-neutral-200">
                {i + 1}. {t.name}
              </div>
              <div className="p-3 space-y-2">
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-neutral-400 mb-0.5">Args</div>
                  <pre className="bg-neutral-50 border border-neutral-200 rounded p-2 overflow-x-auto text-[11px] leading-relaxed font-mono whitespace-pre-wrap break-words">
                    {formatJson(t.args)}
                  </pre>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-neutral-400 mb-0.5">Result</div>
                  <pre className="bg-neutral-50 border border-neutral-200 rounded p-2 overflow-x-auto text-[11px] leading-relaxed font-mono whitespace-pre-wrap break-words max-h-96 overflow-y-auto">
                    {formatJson(t.output)}
                  </pre>
                </div>
              </div>
            </div>
          ))}
        </div>
      </aside>
    </>
  );
};

type Vote = "a" | "b" | "tie" | "both_bad";

const BattleSide = ({
  label,
  text,
  tools,
  side,
  won,
  running,
  onInspect,
}: {
  label: string;
  text: string;
  tools: BattleData["tools_a"];
  side: "a" | "b";
  won: boolean | null;
  running: boolean;
  onInspect: () => void;
}) => {
  const { setOpenThreadId } = useThreadDrawer();
  // Walk tool outputs just like the normal renderer so [ref:] chips resolve.
  const hints = extractThreadHints(
    tools.map((t) => ({ type: "tool-call" as const, result: t.output })),
  );
  const knownIds = hints.map((h) => h.thread_id);

  const colorCls =
    won === true
      ? "border-emerald-300 bg-emerald-50/40"
      : won === false
      ? "border-neutral-200 bg-neutral-50/40 opacity-70"
      : "border-neutral-200";

  return (
    <div className={`rounded-lg border ${colorCls} p-3 flex flex-col min-w-0`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono text-neutral-500">{label}</span>
        {running ? (
          <span className="text-[10px] text-neutral-400">thinking…</span>
        ) : (
          <button
            type="button"
            onClick={onInspect}
            disabled={tools.length === 0}
            className="text-[10px] text-neutral-400 hover:text-neutral-800 hover:underline disabled:no-underline disabled:cursor-default"
            title="Inspect tool calls"
          >
            {tools.length} tool call{tools.length === 1 ? "" : "s"} {tools.length > 0 ? "↗" : ""}
          </button>
        )}
      </div>
      {running && !text ? (
        <div className="flex-1 flex items-center justify-center py-6 text-xs text-neutral-400">
          <Spinner />
          <span className="ml-2">generating…</span>
        </div>
      ) : (
      <div className="prose prose-sm max-w-none prose-p:my-1.5 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0 prose-headings:my-2 prose-pre:my-2">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          urlTransform={(url) => url}
          components={{
            a: ({ href, children, ...rest }) => {
              if (href?.startsWith(REF_PREFIX)) {
                return (
                  <CitationChip
                    threadId={href.slice(REF_PREFIX.length)}
                    hints={hints}
                    onOpen={setOpenThreadId}
                  />
                );
              }
              return (
                <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline" {...rest}>
                  {children}
                </a>
              );
            },
          }}
        >
          {linkifyRefs(text, knownIds)}
        </ReactMarkdown>
      </div>
      )}
    </div>
  );
};

export const BattleMessage = ({ data }: { data: BattleData }) => {
  const [vote, setVote] = useState<Vote | null>(null);
  const [voting, setVoting] = useState(false);
  const [inspect, setInspect] = useState<"a" | "b" | null>(null);
  const anyRunning = !!(data.running_a || data.running_b);

  const submitVote = async (winner: Vote) => {
    if (vote || voting) return;
    setVoting(true);
    try {
      await fetch("/api/battle/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: data.question,
          variant_a: data.variant_a,
          variant_b: data.variant_b,
          winner,
          request_id_a: data.request_id,
          request_id_b: data.request_id,
        }),
      });
      setVote(winner);
    } catch (e) {
      console.error(e);
    } finally {
      setVoting(false);
    }
  };

  const revealed = vote !== null;
  const labelA = revealed ? variantLabel(data.variant_a) : "A";
  const labelB = revealed ? variantLabel(data.variant_b) : "B";
  const wonA = vote === "a" ? true : vote === "b" ? false : null;
  const wonB = vote === "b" ? true : vote === "a" ? false : null;

  const btn = (v: Vote, label: string, hoverCls = "hover:bg-blue-50") => (
    <button
      key={v}
      type="button"
      disabled={revealed || voting || anyRunning}
      onClick={() => submitVote(v)}
      className={`rounded-md border border-neutral-300 px-3 py-1 text-xs ${hoverCls} disabled:opacity-40 disabled:cursor-not-allowed`}
    >
      {label}
    </button>
  );

  return (
    <div className="my-3">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <BattleSide
          label={labelA}
          text={data.answer_a}
          tools={data.tools_a}
          side="a"
          won={wonA}
          running={!!data.running_a}
          onInspect={() => setInspect("a")}
        />
        <BattleSide
          label={labelB}
          text={data.answer_b}
          tools={data.tools_b}
          side="b"
          won={wonB}
          running={!!data.running_b}
          onInspect={() => setInspect("b")}
        />
      </div>
      <div className="mt-2 flex items-center gap-2 flex-wrap">
        {revealed ? (
          <span className="text-xs text-emerald-700">
            ✓ recorded:{" "}
            {vote === "tie"
              ? "tie"
              : vote === "both_bad"
              ? "both bad"
              : vote === "a"
              ? `A wins — ${variantLabel(data.variant_a)}`
              : `B wins — ${variantLabel(data.variant_b)}`}
          </span>
        ) : (
          <>
            <span className="text-xs text-neutral-500">which is better?</span>
            {btn("a", "A wins")}
            {btn("b", "B wins")}
            {btn("tie", "Same", "hover:bg-neutral-50")}
            {btn("both_bad", "Both bad", "hover:bg-red-50")}
          </>
        )}
      </div>
      <InspectorDrawer
        open={inspect !== null}
        onClose={() => setInspect(null)}
        label={inspect === "a" ? labelA : inspect === "b" ? labelB : ""}
        tools={inspect === "a" ? data.tools_a : inspect === "b" ? data.tools_b : []}
      />
    </div>
  );
};
