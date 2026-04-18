"use client";

import { useState } from "react";

import { extractThreadHints } from "@/lib/extractThreadHints";
import { variantLabel, type BattleVariant } from "@/lib/battleVariants";

import { CitableMarkdown } from "./CitableMarkdown";
import { Drawer } from "./Drawer";

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
}) => (
  <Drawer
    open={open}
    onClose={onClose}
    title={`Inspector — ${label}`}
    subtitle={`${tools.length} tool call${tools.length === 1 ? "" : "s"}`}
    widthClass="w-full sm:w-[560px]"
  >
    <div className="px-5 py-3 space-y-4">
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
  </Drawer>
);

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
  // Walk tool outputs just like the normal renderer so [ref:] chips resolve.
  const hints = extractThreadHints(
    tools.map((t) => ({ type: "tool-call" as const, result: t.output })),
  );

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
        <CitableMarkdown text={text} hints={hints} />
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
