"use client";

import { useState } from "react";

// Renderer for `data-deep-stage` parts emitted by the chat route's
// deep-mode translator. Each part carries { kind, payload } where
// kind is one of plan / evidence / tool_call / analysis / skipped
// / code_run / draft / critique / revision / cost.
//
// Compact by default: one line with the stage label + a short
// summary. Click to expand for the raw payload (useful for
// debugging prompt / tool arg issues during development).
//
// The "final" answer text does NOT flow through this component —
// it's a regular text part so CitableMarkdown renders citations.

type Props = {
  data: { kind: string; payload?: unknown };
};

const LABELS: Record<string, { emoji: string; label: string; tint: string }> = {
  plan: { emoji: "🧭", label: "Plan", tint: "border-indigo-200 bg-indigo-50 text-indigo-900" },
  evidence: {
    emoji: "🔎",
    label: "Retrieved",
    tint: "border-sky-200 bg-sky-50 text-sky-900",
  },
  tool_call: {
    emoji: "🛠",
    label: "Tool",
    tint: "border-neutral-200 bg-neutral-50 text-neutral-700",
  },
  analysis: {
    emoji: "📊",
    label: "Analyst",
    tint: "border-emerald-200 bg-emerald-50 text-emerald-900",
  },
  skipped: {
    emoji: "⏭",
    label: "Analyst skipped",
    tint: "border-neutral-200 bg-neutral-50 text-neutral-500",
  },
  code_run: {
    emoji: "💻",
    label: "Code",
    tint: "border-emerald-200 bg-emerald-50 text-emerald-900",
  },
  draft: { emoji: "✏️", label: "Draft", tint: "border-amber-200 bg-amber-50 text-amber-900" },
  critique: {
    emoji: "🧪",
    label: "Critic",
    tint: "border-violet-200 bg-violet-50 text-violet-900",
  },
  revision: { emoji: "🔁", label: "Revision", tint: "border-amber-200 bg-amber-50 text-amber-900" },
  cost: { emoji: "💵", label: "Cost", tint: "border-neutral-200 bg-neutral-50 text-neutral-700" },
};

const summarize = (kind: string, payload: unknown): string => {
  if (!payload || typeof payload !== "object") return "";
  const p = payload as Record<string, unknown>;
  if (kind === "plan") {
    const plan = p.plan as Record<string, unknown> | undefined;
    if (plan) {
      const qt = plan.question_type as string | undefined;
      const r = Array.isArray(plan.retrieval) ? (plan.retrieval as unknown[]).length : 0;
      const a = Array.isArray(plan.analysis) ? (plan.analysis as unknown[]).length : 0;
      return `${qt ?? ""}${qt ? " · " : ""}${r} retrieval · ${a} analysis`;
    }
  }
  if (kind === "evidence") {
    const s = p.summary as string | undefined;
    return s ? s.split("\n")[0] : "";
  }
  if (kind === "tool_call") {
    const name = p.name as string | undefined;
    const args = p.args as Record<string, unknown> | undefined;
    if (args) {
      const argStr = Object.entries(args)
        .map(([k, v]) => `${k}=${JSON.stringify(v).slice(0, 40)}`)
        .join(", ");
      return `${name ?? ""}(${argStr})`;
    }
    return name ?? "";
  }
  if (kind === "critique") {
    const accepted = p.accepted as boolean | undefined;
    const n = Array.isArray(p.violations) ? (p.violations as unknown[]).length : 0;
    return accepted ? "accepted" : `rejected · ${n} violation${n === 1 ? "" : "s"}`;
  }
  if (kind === "cost") {
    const usd = p.usd as number | undefined;
    const total = p.turn_total_usd as number | undefined;
    if (usd !== undefined) return `$${usd.toFixed(5)} (turn total $${(total ?? 0).toFixed(5)})`;
  }
  if (kind === "skipped") {
    return (p.reason as string) ?? "";
  }
  if (kind === "draft" || kind === "revision") {
    const t = p.text as string | undefined;
    return t ? t.slice(0, 80) + (t.length > 80 ? "…" : "") : "";
  }
  return "";
};

export const DeepStagePart = ({ data }: Props) => {
  const kind = data.kind;
  const meta = LABELS[kind] ?? {
    emoji: "•",
    label: kind,
    tint: "border-neutral-200 bg-neutral-50 text-neutral-700",
  };
  const [open, setOpen] = useState(false);
  const summary = summarize(kind, data.payload);
  return (
    <button
      type="button"
      onClick={() => setOpen((v) => !v)}
      className={`mb-1 block w-full rounded border px-2 py-1 text-left text-[11px] ${meta.tint}`}
    >
      <span className="font-medium">
        {meta.emoji} {meta.label}
      </span>
      {summary && <span className="ml-2 text-neutral-500">{summary}</span>}
      {open && (
        <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-white/60 p-1 font-mono text-[10px] leading-tight">
          {JSON.stringify(data.payload, null, 2)}
        </pre>
      )}
    </button>
  );
};
