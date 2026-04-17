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

const collectFromOutput = (result: unknown, into: Map<string, ThreadHint>) => {
  if (!result || typeof result !== "object") return;
  const o = result as Record<string, unknown>;
  const threads = (o.threads ?? o.messages) as unknown;
  if (Array.isArray(threads)) {
    for (const t of threads) {
      if (isThread(t) && typeof t.thread_id === "string") {
        const existing = into.get(t.thread_id);
        if (!existing) {
          into.set(t.thread_id, {
            thread_id: t.thread_id,
            subject: t.subject,
            participants: t.participants,
          });
        }
      }
    }
  }
  // get_thread returns { thread_id, messages: [{ subject, from_addr, ... }] }
  if (typeof o.thread_id === "string" && Array.isArray(o.messages) && o.messages.length > 0) {
    const first = o.messages[0] as { subject?: string; from_addr?: string };
    if (!into.has(o.thread_id)) {
      into.set(o.thread_id, {
        thread_id: o.thread_id,
        subject: first.subject,
        participants: first.from_addr ? [first.from_addr] : undefined,
      });
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
