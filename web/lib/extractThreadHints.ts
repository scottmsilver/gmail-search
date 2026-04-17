import type { ThreadHint } from "@/components/CitationChip";

type AnyMessagePart = {
  type: string;
  toolName?: string;
  result?: unknown;
};

type ToolThread = {
  thread_id?: string;
  subject?: string;
  participants?: string[];
};

const isThread = (v: unknown): v is ToolThread =>
  typeof v === "object" && v !== null && "thread_id" in v;

// Walks an arbitrary tool result tree adding any thread it finds. Handles:
//   { threads: [...] }                       (search/query result row)
//   { thread_id, messages: [...] }           (get_thread result row)
//   { thread_id, subject }                   (validate_reference row)
//   { results: [...above shapes...] }        (batched tool envelope)
const collectFromOutput = (value: unknown, into: Map<string, ThreadHint>): void => {
  if (Array.isArray(value)) {
    for (const v of value) collectFromOutput(v, into);
    return;
  }
  if (!value || typeof value !== "object") return;
  const o = value as Record<string, unknown>;

  if (typeof o.thread_id === "string" && !into.has(o.thread_id)) {
    const fromMessage =
      Array.isArray(o.messages) && o.messages.length > 0
        ? (o.messages[0] as { subject?: string; from_addr?: string })
        : null;
    into.set(o.thread_id, {
      thread_id: o.thread_id,
      subject:
        (typeof o.subject === "string" ? o.subject : undefined) ??
        fromMessage?.subject,
      participants:
        (Array.isArray(o.participants) ? (o.participants as string[]) : undefined) ??
        (fromMessage?.from_addr ? [fromMessage.from_addr] : undefined),
    });
  }

  // Recurse into nested arrays/objects so batched envelopes are walked.
  for (const v of Object.values(o)) {
    if (v && (Array.isArray(v) || typeof v === "object")) {
      collectFromOutput(v, into);
    }
  }
};

export const extractThreadHints = (parts: readonly AnyMessagePart[]): ThreadHint[] => {
  const hints = new Map<string, ThreadHint>();
  for (const p of parts) {
    if (p.type === "tool-call") {
      collectFromOutput(p.result, hints);
    }
  }
  return Array.from(hints.values());
};
