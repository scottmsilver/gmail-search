"use client";

import { useState } from "react";
import type { ToolCallMessagePartProps } from "@assistant-ui/react";

const summarizeArgs = (args: Record<string, unknown> | undefined): string => {
  if (!args) return "";
  const entries = Object.entries(args).filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (entries.length === 0) return "";
  return entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
};

const summarizeOne = (toolName: string, o: Record<string, unknown>): string => {
  if ("error" in o) return `error: ${String(o.error).slice(0, 60)}`;
  if (Array.isArray(o.threads)) return `${o.threads.length} threads`;
  if (Array.isArray(o.messages)) {
    const att = (o.messages as { attachments?: unknown[] }[]).reduce(
      (acc, m) => acc + (m.attachments?.length ?? 0),
      0,
    );
    return `${o.messages.length} msgs / ${att} attachments`;
  }
  if (toolName === "get_attachment_text" && typeof o.extracted_text === "string") {
    return `${String(o.filename ?? "attachment")} (${o.extracted_text.length} chars)`;
  }
  if (typeof o.filename === "string" && typeof o.sizeBytes === "number") {
    return `${o.filename} inlined (${(o.sizeBytes / 1024).toFixed(0)}KB)`;
  }
  if ("valid" in o) return o.valid ? `✓ ${String(o.subject ?? o.thread_id ?? "valid")}` : `✗ ${String(o.error ?? "invalid")}`;
  return "done";
};

const summarizeResult = (toolName: string, result: unknown): string => {
  if (result === undefined) return "running…";
  if (result === null || typeof result !== "object") return "done";
  const o = result as Record<string, unknown>;
  if ("error" in o && !("results" in o)) return `error: ${String(o.error).slice(0, 60)}`;
  if (Array.isArray(o.results)) {
    const items = o.results as Record<string, unknown>[];
    if (items.length === 1) return summarizeOne(toolName, items[0]);
    const errs = items.filter((r) => "error" in r).length;
    return `${items.length} items${errs ? ` (${errs} failed)` : ""}`;
  }
  return summarizeOne(toolName, o);
};

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

const hasError = (result: unknown): boolean => {
  if (!result || typeof result !== "object") return false;
  return "error" in (result as Record<string, unknown>);
};

export const ToolCallUI = ({ toolName, args, result }: ToolCallMessagePartProps) => {
  const errored = hasError(result);
  const [open, setOpen] = useState(errored);
  const argSummary = summarizeArgs(args as Record<string, unknown>);
  const summary = summarizeResult(toolName, result);

  const headerCls = errored
    ? "w-full text-left flex items-start gap-1.5 text-red-700 font-mono hover:text-red-900"
    : "w-full text-left flex items-start gap-1.5 text-neutral-500 font-mono hover:text-neutral-800";
  const nameCls = errored ? "text-red-800 font-semibold" : "text-neutral-700";
  const argCls = errored ? "text-red-500" : "text-neutral-400";

  return (
    <div className="my-1 text-xs">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={headerCls}
        title={open ? "Hide details" : "Show details"}
      >
        <Caret open={open} />
        <span className="truncate flex-1">
          <span className={nameCls}>{toolName}</span>
          {argSummary && <span className={argCls}>({argSummary})</span>}
          <span className={argCls}> → {summary}</span>
        </span>
      </button>
      {open && (
        <div className="mt-1 ml-4 border-l-2 border-neutral-200 pl-3 space-y-2">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-neutral-400 mb-0.5">Arguments</div>
            <pre className="bg-neutral-50 border border-neutral-200 rounded p-2 overflow-x-auto text-[11px] leading-relaxed font-mono whitespace-pre-wrap break-words">
              {formatJson(args)}
            </pre>
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-wide text-neutral-400 mb-0.5">Result</div>
            <pre className="bg-neutral-50 border border-neutral-200 rounded p-2 overflow-x-auto text-[11px] leading-relaxed font-mono whitespace-pre-wrap break-words max-h-96 overflow-y-auto">
              {result === undefined ? "(running…)" : formatJson(result)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
