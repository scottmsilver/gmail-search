import { dynamicTool, tool } from "ai";
import { z } from "zod";

import {
  getAttachmentBytesBackend,
  getAttachmentTextBackend,
  getMessageBackend,
  getThreadBackend,
  lookupThreadByCiteRef,
  queryEmailsBackend,
  searchEmailsBackend,
} from "./backend";

const MAX_INLINE_BYTES = 15 * 1024 * 1024;
// Aggregate cap across an entire batch of binaries. Without this, the
// model can call get_attachment_pdf with 10 attachment_ids × 15 MB each
// and we ship 150 MB of base64 to Gemini in one request — which fails
// noisily if it doesn't OOM Node first.
const MAX_BATCH_INLINE_BYTES = 25 * 1024 * 1024;
// Binary tools cap their batch tighter than text tools.
const MAX_BINARY_BATCH = 4;
const ENRICH_TOP_N = 3;
const ENRICHED_BODY_CHARS = 1500;
const MAX_BATCH = 10;

// ────────────────────────────────────────────────────────────────────
// shared helpers
// ────────────────────────────────────────────────────────────────────

const safely = async <T>(name: string, fn: () => Promise<T>): Promise<T | { error: string }> => {
  try {
    return await fn();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[tool:${name}] failed:`, err);
    return { error: `${name} failed: ${message}` };
  }
};

const citeRef = (threadId: string): string => threadId.slice(0, 8);

// Run an array of items in parallel. Each item gets its own try/catch so
// one failure doesn't poison the rest. The input is echoed alongside the
// output so the model can match results to inputs by index/key.
const runBatch = async <I, O extends Record<string, unknown>>(
  name: string,
  items: I[],
  itemKey: (i: I) => Record<string, unknown>,
  fn: (item: I) => Promise<O>,
): Promise<{ results: Array<Record<string, unknown>> }> => {
  const settled = await Promise.allSettled(items.map(fn));
  const results: Array<Record<string, unknown>> = settled.map((s, idx) => {
    const key = itemKey(items[idx]);
    if (s.status === "fulfilled") {
      return { ...key, ...s.value };
    }
    const err = s.reason instanceof Error ? s.reason.message : String(s.reason);
    console.error(`[tool:${name}] item ${idx} failed:`, s.reason);
    return { ...key, error: err };
  });
  return { results };
};

// Run a batch of binary attachment fetches with both per-item and
// aggregate-byte caps. Anything past the aggregate cap is returned as
// an error result so the model knows to retry with fewer items.
const runBatchedBinary = async (
  name: string,
  attachmentIds: number[],
  mimeOk: (att: Awaited<ReturnType<typeof getAttachmentBytesBackend>>) => boolean,
  expectedMime: string,
): Promise<{ results: Array<Record<string, unknown>> }> => {
  const settled = await Promise.allSettled(attachmentIds.map(getAttachmentBytesBackend));
  let bytesUsed = 0;
  const results: Array<Record<string, unknown>> = settled.map((s, idx) => {
    const id = attachmentIds[idx];
    if (s.status === "rejected") {
      const err = s.reason instanceof Error ? s.reason.message : String(s.reason);
      console.error(`[tool:${name}] item ${idx} failed:`, s.reason);
      return { attachment_id: id, error: err };
    }
    const att = s.value;
    if (!mimeOk(att)) {
      return {
        attachment_id: id,
        error: `Attachment is ${att.mimeType}, not ${expectedMime}.`,
      };
    }
    if (att.sizeBytes > MAX_INLINE_BYTES) {
      return {
        attachment_id: id,
        error: `Attachment is ${(att.sizeBytes / 1024 / 1024).toFixed(1)}MB — exceeds per-item cap (${(MAX_INLINE_BYTES / 1024 / 1024).toFixed(0)}MB).`,
      };
    }
    if (bytesUsed + att.sizeBytes > MAX_BATCH_INLINE_BYTES) {
      return {
        attachment_id: id,
        error: `Skipped to stay under ${(MAX_BATCH_INLINE_BYTES / 1024 / 1024).toFixed(0)}MB batch cap. Call again with fewer items.`,
      };
    }
    bytesUsed += att.sizeBytes;
    return {
      attachment_id: id,
      filename: att.filename,
      mimeType: att.mimeType,
      sizeBytes: att.sizeBytes,
      base64: att.base64,
    };
  });
  return { results };
};

// ────────────────────────────────────────────────────────────────────
// search_emails formatters
// ────────────────────────────────────────────────────────────────────

type RawSearchResults = Awaited<ReturnType<typeof searchEmailsBackend>>;

const baseSearchEntry = (t: RawSearchResults[number]) => {
  const top = t.matches[0];
  return {
    thread_id: t.thread_id,
    cite_ref: citeRef(t.thread_id),
    subject: t.subject,
    participants: t.participants,
    message_count: t.message_count,
    date_last: t.date_last,
    from_addr: top?.from_addr ?? "",
    snippet: top?.snippet ?? "",
    score: Number(t.score.toFixed(3)),
    top_message_id: top?.message_id ?? null,
  };
};

const fetchTopBody = async (messageId: string | null): Promise<string | null> => {
  if (!messageId) return null;
  try {
    const msg = await getMessageBackend(messageId);
    return (msg.body_text ?? "").slice(0, ENRICHED_BODY_CHARS);
  } catch {
    return null;
  }
};

const enrichTopHits = async (entries: ReturnType<typeof baseSearchEntry>[]) => {
  const heads = entries.slice(0, ENRICH_TOP_N);
  const tail = entries.slice(ENRICH_TOP_N);
  const enriched = await Promise.all(
    heads.map(async (e) => ({ ...e, body_excerpt: await fetchTopBody(e.top_message_id) })),
  );
  return [...enriched, ...tail];
};

const formatSearchOutput = async (raw: RawSearchResults) => enrichTopHits(raw.map(baseSearchEntry));

const formatQueryOutput = (raw: Awaited<ReturnType<typeof queryEmailsBackend>>) =>
  raw.map((t) => ({
    thread_id: t.thread_id,
    cite_ref: citeRef(t.thread_id),
    subject: t.subject,
    participants: t.participants,
    message_count: t.message_count,
    date_last: t.date_last,
    snippet: t.snippet,
  }));

// ────────────────────────────────────────────────────────────────────
// schemas (single item per row — tools take an array of these)
// ────────────────────────────────────────────────────────────────────

const SearchSchema = z.object({
  query: z.string().describe("Natural language query."),
  top_k: z.number().int().min(1).max(20).optional().describe("Threads to return (default 10)."),
});

const QuerySchema = z.object({
  sender: z.string().optional().describe("Substring match on from address."),
  subject_contains: z.string().optional(),
  date_from: z.string().optional().describe("ISO date YYYY-MM-DD, inclusive."),
  date_to: z.string().optional().describe("ISO date YYYY-MM-DD, inclusive."),
  label: z.string().optional().describe("Gmail label (INBOX, IMPORTANT, SENT, UNREAD)."),
  has_attachment: z.boolean().optional(),
  order_by: z.enum(["date_desc", "date_asc"]).optional(),
  limit: z.number().int().min(1).max(100).optional().describe("Default 20, max 100."),
});

// ────────────────────────────────────────────────────────────────────
// tools
// ────────────────────────────────────────────────────────────────────

export const buildTools = () => ({
  search_emails: tool({
    description:
      "Semantic + keyword hybrid search over the user's Gmail archive. BATCH: pass one OR many queries in `searches` and they run in parallel — prefer ONE call with multiple searches over multiple sequential calls. Each entry returns its own ranked threads. Use for any question where relevance to a topic matters. Returns short snippets — call get_thread for full content.",
    inputSchema: z.object({
      searches: z.array(SearchSchema).min(1).max(MAX_BATCH).describe("One or more queries to run in parallel."),
    }),
    execute: async ({ searches }) =>
      safely("search_emails", () =>
        runBatch(
          "search_emails",
          searches,
          (s) => ({ query: s.query }),
          async (s) => {
            const raw = await searchEmailsBackend({ query: s.query, top_k: s.top_k });
            return { threads: await formatSearchOutput(raw) };
          },
        ),
      ),
  }),

  query_emails: tool({
    description:
      "Structured filter over Gmail by metadata. BATCH: pass one OR many filter sets in `queries` and they run in parallel — prefer ONE call with multiple filters over multiple sequential calls. Useful for comparisons (e.g. emails-per-month). Returns short snippets — call get_thread for full content.",
    inputSchema: z.object({
      queries: z.array(QuerySchema).min(1).max(MAX_BATCH).describe("One or more filter sets to run in parallel."),
    }),
    execute: async ({ queries }) =>
      safely("query_emails", () =>
        runBatch(
          "query_emails",
          queries,
          (q) => ({ filter: q }),
          async (q) => {
            const raw = await queryEmailsBackend(q);
            return { threads: formatQueryOutput(raw) };
          },
        ),
      ),
  }),

  validate_reference: tool({
    description:
      "Verify cite_refs before citing them. BATCH: pass one OR many `cite_refs` and they resolve in parallel — prefer ONE call with all refs you want to validate over multiple sequential calls. Each entry returns whether the prefix matches a real thread.",
    inputSchema: z.object({
      cite_refs: z
        .array(z.string().min(4).max(20))
        .min(1)
        .max(MAX_BATCH)
        .describe("Hex cite_refs (each at least 4 hex chars)."),
    }),
    execute: async ({ cite_refs }) =>
      safely("validate_reference", () =>
        runBatch(
          "validate_reference",
          cite_refs,
          (r) => ({ cite_ref: r }),
          async (r) => {
            const res = await lookupThreadByCiteRef(r);
            if (res.ok) {
              return {
                valid: true,
                thread_id: res.hit.thread_id,
                subject: res.hit.subject,
              };
            }
            return { valid: false, error: res.error, candidates: res.candidates };
          },
        ),
      ),
  }),

  get_thread: tool({
    description:
      "Fetch every message in one or more threads with full body text + attachments. BATCH: pass one OR many `thread_ids` and they fetch in parallel — prefer ONE call with all the threads you need over multiple sequential calls. Use this after search/query when snippets aren't enough.",
    inputSchema: z.object({
      thread_ids: z
        .array(z.string())
        .min(1)
        .max(MAX_BATCH)
        .describe("Thread IDs from search/query results."),
    }),
    execute: async ({ thread_ids }) =>
      safely("get_thread", () =>
        runBatch(
          "get_thread",
          thread_ids,
          (t) => ({ thread_id: t }),
          async (t) => {
            const detail = await getThreadBackend(t);
            return {
              cite_ref: citeRef(detail.thread_id),
              messages: detail.messages.map((m) => ({
                message_id: m.id,
                from_addr: m.from_addr,
                to_addr: m.to_addr,
                date: m.date,
                subject: m.subject,
                body_text: m.body_text,
                attachments: m.attachments,
              })),
            };
          },
        ),
      ),
  }),

  get_attachment_text: tool({
    description:
      "Return text already extracted from one or more attachments (PDFs, docx, csv, calendar invites, etc.). BATCH: pass one OR many `attachment_ids` — prefer ONE call with all the attachments you want over multiple sequential calls. Cheap — use first when you only need the words.",
    inputSchema: z.object({
      attachment_ids: z
        .array(z.number().int())
        .min(1)
        .max(MAX_BATCH)
        .describe("Attachment IDs from get_thread."),
    }),
    execute: async ({ attachment_ids }) =>
      safely("get_attachment_text", () =>
        runBatch(
          "get_attachment_text",
          attachment_ids,
          (id) => ({ attachment_id: id }),
          (id) => getAttachmentTextBackend(id),
        ),
      ),
  }),

  get_attachment_image: dynamicTool({
    description:
      "Inline one or more image attachments directly so you can see them. BATCH: pass `attachment_ids` array — prefer ONE call with all images you need to see over multiple sequential calls. Only use for image/* attachments and only when visual content matters. Limit ~4 per batch.",
    inputSchema: z.object({
      attachment_ids: z.array(z.number().int()).min(1).max(MAX_BINARY_BATCH),
    }),
    execute: async (input) =>
      safely("get_attachment_image", async () => {
        const { attachment_ids } = input as { attachment_ids: number[] };
        return runBatchedBinary(
          "get_attachment_image",
          attachment_ids,
          (att) => att.mimeType.startsWith("image/"),
          "image/*",
        );
      }),
    toModelOutput: ({ output }) => {
      const o = output as { results?: Array<Record<string, unknown>> } | { error?: string };
      if (!o || typeof o !== "object" || "error" in o) {
        return { type: "json", value: { error: String((o as { error?: string })?.error ?? "unknown") } };
      }
      const results = (o as { results: Array<Record<string, unknown>> }).results;
      const value: Array<{ type: "text"; text: string } | { type: "image-data"; data: string; mediaType: string }> = [];
      for (const r of results) {
        if (r.error || !r.base64) {
          value.push({ type: "text", text: `Image attachment ${r.attachment_id} skipped: ${String(r.error ?? "no data")}` });
          continue;
        }
        value.push({ type: "text", text: `Inlined image: ${String(r.filename)}` });
        value.push({ type: "image-data", data: String(r.base64), mediaType: String(r.mimeType) });
      }
      return { type: "content", value };
    },
  }),

  get_attachment_pdf: dynamicTool({
    description:
      "Inline one or more PDF attachments directly for layout-aware reading. BATCH: pass `attachment_ids` array — prefer ONE call with all PDFs you need to see over multiple sequential calls. Prefer get_attachment_text first; only escalate to PDF when layout matters. Limit ~4 per batch.",
    inputSchema: z.object({
      attachment_ids: z.array(z.number().int()).min(1).max(MAX_BINARY_BATCH),
    }),
    execute: async (input) =>
      safely("get_attachment_pdf", async () => {
        const { attachment_ids } = input as { attachment_ids: number[] };
        return runBatchedBinary(
          "get_attachment_pdf",
          attachment_ids,
          (att) => att.mimeType === "application/pdf",
          "application/pdf",
        );
      }),
    toModelOutput: ({ output }) => {
      const o = output as { results?: Array<Record<string, unknown>> } | { error?: string };
      if (!o || typeof o !== "object" || "error" in o) {
        return { type: "json", value: { error: String((o as { error?: string })?.error ?? "unknown") } };
      }
      const results = (o as { results: Array<Record<string, unknown>> }).results;
      const value: Array<
        | { type: "text"; text: string }
        | { type: "file-data"; data: string; mediaType: string; filename: string }
      > = [];
      for (const r of results) {
        if (r.error || !r.base64) {
          value.push({ type: "text", text: `PDF attachment ${r.attachment_id} skipped: ${String(r.error ?? "no data")}` });
          continue;
        }
        value.push({ type: "text", text: `Inlined PDF: ${String(r.filename)}` });
        value.push({
          type: "file-data",
          data: String(r.base64),
          mediaType: String(r.mimeType),
          filename: String(r.filename),
        });
      }
      return { type: "content", value };
    },
  }),
});

export type AppTools = ReturnType<typeof buildTools>;
