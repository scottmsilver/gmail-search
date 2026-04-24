import type { AttachmentHint } from "@/components/AttachmentChip";
import type { ThreadHint } from "@/components/CitationChip";

type AnyMessagePart = {
  type: string;
  toolName?: string;
  result?: unknown;
};

export type CitationHints = {
  threads: ThreadHint[];
  attachments: AttachmentHint[];
};

// Walks an arbitrary tool result tree gathering citation-chip metadata.
// Handles the shapes we actually emit:
//   { threads: [...] }                              search / query result row
//   { thread_id, messages: [{ attachments: [...]}]} get_thread row
//   { thread_id, subject }                          validate_reference row
//   { results: [ { attachment_id, filename, ... }]} get_attachment_* rows
//   { results: [...above shapes...] }               batched tool envelope
const collect = (
  value: unknown,
  threadsOut: Map<string, ThreadHint>,
  attsOut: Map<number, AttachmentHint>,
  currentThreadId: string | null = null,
  // True when the walker is descending into the `attachments` field of
  // a parent row — tells us a nested `{id, filename, ...}` is genuinely
  // an attachment, not some other shape that happens to carry both.
  inAttachmentList = false,
): void => {
  if (Array.isArray(value)) {
    for (const v of value) collect(v, threadsOut, attsOut, currentThreadId, inAttachmentList);
    return;
  }
  if (!value || typeof value !== "object") return;
  const o = value as Record<string, unknown>;

  // Thread hint — also tracks thread_id so nested attachments can adopt it.
  let threadIdHere = currentThreadId;
  if (typeof o.thread_id === "string") {
    threadIdHere = o.thread_id;
    if (!threadsOut.has(o.thread_id)) {
      const fromMessage =
        Array.isArray(o.messages) && o.messages.length > 0
          ? (o.messages[0] as { subject?: string; from_addr?: string })
          : null;
      threadsOut.set(o.thread_id, {
        thread_id: o.thread_id,
        subject:
          (typeof o.subject === "string" ? o.subject : undefined) ??
          fromMessage?.subject,
        participants:
          (Array.isArray(o.participants) ? (o.participants as string[]) : undefined) ??
          (fromMessage?.from_addr ? [fromMessage.from_addr] : undefined),
      });
    }
  }

  // Attachment hint. Match ONLY explicit `attachment_id` — a bare
  // `id: 5` + `filename: "..."` shape could be anything (a search
  // result, a conversation row, a message snippet). The `get_thread`
  // path nests attachment rows with `{id, filename, mime_type, ...}`
  // — we surface those via a second match-by-id below, but only when
  // the parent row's `attachments` field is what walked us here.
  const attId =
    typeof o.attachment_id === "number"
      ? o.attachment_id
      : inAttachmentList && typeof o.id === "number"
        ? o.id
        : null;
  if (attId !== null && !attsOut.has(attId)) {
    attsOut.set(attId, {
      attachment_id: attId,
      filename: typeof o.filename === "string" ? o.filename : undefined,
      mime_type:
        (typeof o.mime_type === "string" ? o.mime_type : undefined) ??
        (typeof o.mimeType === "string" ? o.mimeType : undefined),
      thread_id: threadIdHere ?? undefined,
    });
  }

  for (const [k, v] of Object.entries(o)) {
    if (v && (Array.isArray(v) || typeof v === "object")) {
      // Flag descent INTO an `attachments` field so inner `{id, ...}`
      // rows can be treated as attachments without needing
      // `attachment_id` explicitly.
      collect(v, threadsOut, attsOut, threadIdHere, k === "attachments");
    }
  }
};

export const extractThreadHints = (parts: readonly AnyMessagePart[]): ThreadHint[] => {
  return extractCitationHints(parts).threads;
};

export const extractCitationHints = (parts: readonly AnyMessagePart[]): CitationHints => {
  const threads = new Map<string, ThreadHint>();
  const attachments = new Map<number, AttachmentHint>();
  for (const p of parts) {
    if (p.type === "tool-call") {
      collect(p.result, threads, attachments);
    }
  }
  return {
    threads: Array.from(threads.values()),
    attachments: Array.from(attachments.values()),
  };
};
