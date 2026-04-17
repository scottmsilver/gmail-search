"use client";

import type { ToolCallMessagePartProps } from "@assistant-ui/react";

const summarizeArgs = (args: Record<string, unknown> | undefined): string => {
  if (!args) return "";
  const entries = Object.entries(args).filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (entries.length === 0) return "";
  return entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
};

const summarizeResult = (toolName: string, result: unknown): string => {
  if (result === undefined) return "running…";
  if (result === null || typeof result !== "object") return "done";
  const o = result as Record<string, unknown>;
  if ("error" in o) return `error: ${String(o.error).slice(0, 60)}`;
  if (Array.isArray(o.threads)) return `${o.threads.length} threads`;
  if (Array.isArray(o.messages)) {
    const att = (o.messages as { attachments?: unknown[] }[]).reduce(
      (acc, m) => acc + (m.attachments?.length ?? 0),
      0,
    );
    return `${o.messages.length} messages, ${att} attachments`;
  }
  if (toolName === "get_attachment_text" && typeof o.extracted_text === "string") {
    return `${o.filename ?? "attachment"} — ${o.extracted_text.length} chars`;
  }
  if (typeof o.filename === "string" && typeof o.sizeBytes === "number") {
    return `${o.filename} inlined (${(o.sizeBytes / 1024).toFixed(0)}KB)`;
  }
  return "done";
};

export const ToolCallUI = ({ toolName, args, result }: ToolCallMessagePartProps) => {
  const argSummary = summarizeArgs(args as Record<string, unknown>);
  return (
    <div className="my-1 text-xs text-neutral-500 font-mono flex items-start gap-1.5">
      <span className="text-neutral-400">↳</span>
      <span className="truncate">
        <span className="text-neutral-700">{toolName}</span>
        {argSummary && <span className="text-neutral-400">({argSummary})</span>}
        <span className="text-neutral-400"> → {summarizeResult(toolName, result)}</span>
      </span>
    </div>
  );
};
