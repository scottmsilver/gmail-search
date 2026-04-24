"use client";

import { useState } from "react";
import { useMessage } from "@assistant-ui/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { AttachmentInlineViewer } from "./AttachmentInlineViewer";

type ToolPart = {
  type: "tool-call";
  toolCallId: string;
  toolName: string;
  args: unknown;
  result?: unknown;
  isError?: boolean;
};

type ReasoningPart = { type: "reasoning"; text: string };

type WorkPart = ToolPart | ReasoningPart;

const isToolPart = (p: { type: string }): p is ToolPart => p.type === "tool-call";
const isReasoningPart = (p: { type: string }): p is ReasoningPart => p.type === "reasoning";

const Caret = ({ open }: { open: boolean }) => (
  <svg
    className={`w-3 h-3 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
  </svg>
);

const Spinner = () => (
  <svg className="w-3 h-3 animate-spin shrink-0" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth={3} opacity={0.25} />
    <path d="M22 12a10 10 0 0 1-10 10" stroke="currentColor" strokeWidth={3} strokeLinecap="round" />
  </svg>
);

const stripBinary = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(stripBinary);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (k === "base64" && typeof v === "string") {
        out[k] = `<${v.length} chars base64>`;
      } else {
        out[k] = stripBinary(v);
      }
    }
    return out;
  }
  return value;
};

const formatJson = (value: unknown): string => {
  try {
    return JSON.stringify(stripBinary(value), null, 2);
  } catch {
    return String(value);
  }
};

const summarizeArgs = (args: unknown): string => {
  if (!args || typeof args !== "object") return "";
  const entries = Object.entries(args as Record<string, unknown>).filter(
    ([, v]) => v !== undefined && v !== null && v !== "",
  );
  if (entries.length === 0) return "";
  return entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
};

const summarizeResult = (toolName: string, result: unknown): string => {
  if (result === undefined) return "running…";
  if (result === null || typeof result !== "object") return "done";
  const o = result as Record<string, unknown>;
  if ("error" in o && !("results" in o)) return `error: ${String(o.error).slice(0, 60)}`;
  if (Array.isArray(o.results)) {
    const items = o.results as Record<string, unknown>[];
    const errs = items.filter((r) => "error" in r).length;
    // Attachment-specific summary: show "file.pdf (as=text, 5.4k chars)"
    // etc. so the collapsed header conveys what the model actually got.
    if (toolName === "get_attachment") {
      const parts = items.map((it) => {
        if (it.error) return `${String(it.attachment_id)} err`;
        const as = String(it.as ?? "?");
        const name = String(it.filename ?? `#${it.attachment_id}`);
        if (as === "text") return `${name} (text, ${String(it.text_chars ?? 0)} chars)`;
        if (as === "rendered_pages") {
          const pages = Array.isArray(it.pages) ? it.pages.length : 0;
          return `${name} (${pages} pages)`;
        }
        const kb = it.size_bytes ? ` ${((Number(it.size_bytes) || 0) / 1024).toFixed(0)}KB` : "";
        return `${name} (${as}${kb})`;
      });
      return parts.join(" · ");
    }
    return `${items.length} items${errs ? ` (${errs} failed)` : ""}`;
  }
  if (Array.isArray(o.threads)) return `${o.threads.length} threads`;
  if (Array.isArray(o.messages)) return `${o.messages.length} msgs`;
  if ("valid" in o) return o.valid ? "✓ valid" : `✗ ${String(o.error ?? "invalid")}`;
  return "done";
};

const toolSummary = (tools: ToolPart[]): string => {
  if (tools.length === 0) return "";
  const counts = new Map<string, number>();
  for (const t of tools) counts.set(t.toolName, (counts.get(t.toolName) ?? 0) + 1);
  return Array.from(counts.entries())
    .map(([name, n]) => (n > 1 ? `${n}× ${name}` : name))
    .join(", ");
};

const stripMarkdownForPreview = (raw: string): string =>
  raw
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/[#*_`>~|]/g, "")
    .replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();

const ToolBlock = ({ part }: { part: ToolPart }) => {
  const [open, setOpen] = useState(!!part.isError);
  const errored = part.isError;
  const argSummary = summarizeArgs(part.args);
  const summary = summarizeResult(part.toolName, part.result);
  const headerCls = errored
    ? "w-full text-left flex items-start gap-1.5 text-red-700 font-mono hover:text-red-900"
    : "w-full text-left flex items-start gap-1.5 text-neutral-500 font-mono hover:text-neutral-800";
  const nameCls = errored ? "text-red-800 font-semibold" : "text-neutral-700";
  const argCls = errored ? "text-red-500" : "text-neutral-400";

  return (
    <div className="text-xs">
      <button type="button" onClick={() => setOpen((v) => !v)} className={headerCls}>
        <Caret open={open} />
        <span className="truncate flex-1">
          <span className={nameCls}>{part.toolName}</span>
          {argSummary && <span className={argCls}>({argSummary})</span>}
          <span className={argCls}> → {summary}</span>
        </span>
      </button>
      {open && (
        <div className="mt-1 ml-4 border-l-2 border-neutral-200 pl-3 space-y-2">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-neutral-400 mb-0.5">Arguments</div>
            <pre className="bg-neutral-50 border border-neutral-200 rounded p-2 overflow-x-auto text-[11px] leading-relaxed font-mono whitespace-pre-wrap break-words">
              {formatJson(part.args)}
            </pre>
          </div>
          {/* For get_attachment, show the model's exact view (text block,
              PDF iframe, image, or page rasters) BEFORE the raw JSON so
              the human user can see what the agent saw without mentally
              decoding a base64 payload. Raw JSON stays below as a
              debug fallback but has binary fields stripped. */}
          {part.toolName === "get_attachment" && part.result !== undefined && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-neutral-400 mb-0.5">What the model saw</div>
              <AttachmentInlineViewer result={part.result} />
            </div>
          )}
          <div>
            <div className="text-[11px] uppercase tracking-wide text-neutral-400 mb-0.5">Result (raw)</div>
            <pre className="bg-neutral-50 border border-neutral-200 rounded p-2 overflow-x-auto text-[11px] leading-relaxed font-mono whitespace-pre-wrap break-words max-h-96 overflow-y-auto">
              {part.result === undefined ? "(running…)" : formatJson(part.result)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

const ReasoningBlock = ({ part }: { part: ReasoningPart }) => (
  <div className="text-xs text-neutral-500">
    <div className="flex items-center gap-1.5 italic">
      <span className="text-neutral-400">thoughts</span>
    </div>
    <div className="mt-1 ml-4 border-l-2 border-neutral-200 pl-3 text-[12px] leading-relaxed text-neutral-600 prose prose-sm max-w-none prose-p:my-1 prose-headings:my-1.5 prose-headings:text-neutral-700 prose-strong:text-neutral-700 prose-li:my-0 prose-ul:my-1 prose-ol:my-1">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{part.text}</ReactMarkdown>
    </div>
  </div>
);

// Single disclosure that collapses all tool calls + reasoning for an
// assistant turn into one "Worked for N steps" summary. Keeps vertical
// space small when many tools are called (MAX_TOOL_STEPS=15).
export const AssistantWork = () => {
  const parts = useMessage((m) => m.content) as readonly { type: string }[];
  const isRunning = useMessage((m) => m.status?.type === "running");
  const [open, setOpen] = useState(false);

  const work: WorkPart[] = [];
  for (const p of parts) {
    if (isToolPart(p) || isReasoningPart(p)) work.push(p);
  }
  if (work.length === 0) return null;

  const tools = work.filter(isToolPart);
  const reasoning = work.filter(isReasoningPart);
  const lastReasoning = reasoning[reasoning.length - 1];
  const anyPending = isRunning && tools.some((t) => t.result === undefined);

  let label: string;
  if (anyPending) {
    label = `Working (${tools.length} tool call${tools.length === 1 ? "" : "s"}${reasoning.length ? ", thinking" : ""})`;
  } else if (tools.length === 0) {
    label = "Thoughts";
  } else {
    label = `Worked (${tools.length} tool call${tools.length === 1 ? "" : "s"})`;
  }

  const preview = !open && !anyPending && lastReasoning?.text
    ? stripMarkdownForPreview(lastReasoning.text).slice(0, 70)
    : !open && tools.length > 0
    ? toolSummary(tools)
    : "";

  return (
    <div className="my-1 text-xs text-neutral-500">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 hover:text-neutral-800 max-w-full"
      >
        {anyPending ? <Spinner /> : <Caret open={open} />}
        <span className="italic">{label}</span>
        {preview && (
          <span className="text-neutral-400 truncate max-w-[60ch]">— {preview}{preview.length >= 70 ? "…" : ""}</span>
        )}
      </button>
      {open && (
        <div className="mt-2 ml-4 border-l-2 border-neutral-200 pl-3 space-y-2">
          {work.map((p, i) =>
            isToolPart(p) ? (
              <ToolBlock key={`t-${i}`} part={p} />
            ) : (
              <ReasoningBlock key={`r-${i}`} part={p} />
            ),
          )}
        </div>
      )}
    </div>
  );
};
