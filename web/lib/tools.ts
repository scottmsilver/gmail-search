import { dynamicTool, tool } from "ai";
import { z } from "zod";

import {
  getAttachmentBytesBackend,
  getAttachmentTextBackend,
  getMessageBackend,
  getThreadBackend,
  lookupThreadByCiteRef,
  queryEmailsBackend,
  runSqlBackend,
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
// Enrich the first ENRICH_TOP_N hits with the matching message's body
// text up to ENRICHED_BODY_CHARS. Tail results keep their summary +
// short snippet. Gemini's 1M-token window means we can afford
// generous body excerpts so the model doesn't need to call get_thread
// for simple questions.
const ENRICH_TOP_N = 5;
const ENRICHED_BODY_CHARS = 4000;
// For query_emails (metadata filter) the top matches still benefit
// from some body. Smaller cap since query_emails lists can be up to
// 20 results and we want to keep total payload reasonable.
const QUERY_ENRICH_TOP_N = 3;
const QUERY_ENRICHED_BODY_CHARS = 2000;
const MAX_BATCH = 10;

// Tune: below these thresholds we tell the model the result is thin
// and it should consider broadening. Empirically most useful answers
// have at least 3 hits and top score > 0.45.
const LOW_SCORE_THRESHOLD = 0.45;
const THIN_RESULT_COUNT = 3;

// Returns a quality_note string the model can see, or null if results
// look good. Hints are phrased to trigger the "broaden when unsure"
// rules in the system prompt.
const qualityNoteForSearch = (
  threads: Array<{ score?: number }>,
  query: string,
): string | null => {
  if (threads.length === 0) {
    return `No matches for ${JSON.stringify(query)}. Try alternative phrasings, broader terms, or a related topic — don't claim there are no such emails without trying 1-2 more searches.`;
  }
  const topScore = threads[0]?.score ?? 0;
  if (topScore < LOW_SCORE_THRESHOLD) {
    return `Low confidence — top relevance score ${topScore.toFixed(2)} for ${JSON.stringify(query)}. Results may be off-topic. Consider re-searching with different terms before synthesizing.`;
  }
  if (threads.length < THIN_RESULT_COUNT) {
    return `Only ${threads.length} result(s) for ${JSON.stringify(query)}. If the user is asking about a topic that likely has more emails, try an alternative phrasing.`;
  }
  return null;
};

const qualityNoteForQuery = (threadCount: number, hasFilters: boolean): string | null => {
  if (threadCount === 0) {
    return hasFilters
      ? "No threads matched these filters. The filters may be too narrow, or the data may not exist in this form. Try loosening date_from/date_to, dropping a filter, or using search_emails with natural-language terms."
      : "No threads returned. Consider adding filters or switching to search_emails.";
  }
  return null;
};

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
    // Pre-computed local-model summary of the top-matching message.
    // When present, it's usually enough to answer without calling get_thread.
    summary: top?.summary ?? "",
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

// Fetches the latest message body for a thread to use as body_excerpt
// on query_emails results. Falls back to null on any error.
const fetchLatestBodyForThread = async (threadId: string): Promise<string | null> => {
  try {
    const detail = await getThreadBackend(threadId);
    if (detail.messages.length === 0) return null;
    const latest = detail.messages[detail.messages.length - 1];
    return (latest.body_text ?? "").slice(0, QUERY_ENRICHED_BODY_CHARS);
  } catch {
    return null;
  }
};

const enrichQueryTopHits = async (entries: ReturnType<typeof formatQueryOutput>) => {
  const heads = entries.slice(0, QUERY_ENRICH_TOP_N);
  const tail = entries.slice(QUERY_ENRICH_TOP_N);
  const enriched = await Promise.all(
    heads.map(async (e) => ({ ...e, body_excerpt: await fetchLatestBodyForThread(e.thread_id) })),
  );
  return [...enriched, ...tail];
};

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
      "Semantic + keyword hybrid search over the user's Gmail archive. BATCH: pass one OR many queries in `searches` and they run in parallel — prefer ONE call with multiple searches over multiple sequential calls. Each entry returns its own ranked threads. Every thread carries a `summary` (1-3 sentences from a local model with the full message in hand) and the top 5 threads additionally carry a `body_excerpt` (up to 4000 chars of the matching message). You usually do NOT need get_thread — only call it if the body_excerpt is too short or you need multiple messages from the same thread.",
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
            const threads = await formatSearchOutput(raw);
            const note = qualityNoteForSearch(threads, s.query);
            return note ? { threads, quality_note: note } : { threads };
          },
        ),
      ),
  }),

  query_emails: tool({
    description:
      "Structured filter over Gmail by metadata. BATCH: pass one OR many filter sets in `queries` and they run in parallel — prefer ONE call with multiple filters over multiple sequential calls. Useful for comparisons (e.g. emails-per-month). Every thread carries a summary; the top 3 additionally carry body_excerpt (up to 2000 chars of the latest message). Call get_thread only when you need earlier messages or more than 2000 chars.",
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
            const threads = await enrichQueryTopHits(formatQueryOutput(raw));
            const hasFilters = Object.values(q).some(
              (v) => v !== undefined && v !== null && v !== "" && v !== "date_desc" && v !== 20,
            );
            const note = qualityNoteForQuery(threads.length, hasFilters);
            return note ? { threads, quality_note: note } : { threads };
          },
        ),
      ),
  }),

  sql_query: tool({
    description: `Run a read-only SQL SELECT (or WITH…SELECT) against the local SQLite database. Use this when the other tools can't express the question — aggregations, GROUP BY, OR across fields, relative dates, JOINs, NOT-EXISTS, multi-field filters. Max 500 rows returned, 10s timeout. Read-only enforced at DB + keyword level; INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA are rejected.

BATCH: pass one OR many queries in \`queries\` and they run in parallel.

Key tables:
- messages(id TEXT PK, thread_id TEXT, from_addr TEXT, to_addr TEXT, subject TEXT, body_text TEXT, date TEXT [ISO UTC], labels TEXT [JSON array], history_id INT)
- thread_summary(thread_id TEXT PK, subject TEXT, participants TEXT [JSON], all_from_addrs TEXT [JSON], all_labels TEXT [JSON], message_count INT, date_first TEXT, date_last TEXT)
- attachments(id INT PK, message_id TEXT, filename TEXT, mime_type TEXT, size_bytes INT, extracted_text TEXT)
- topics(topic_id TEXT PK, parent_id TEXT, label TEXT, depth INT, message_count INT)
- message_topics(message_id TEXT, topic_id TEXT)
- contact_frequency(addr TEXT PK, email_count INT, reply_count INT, last_email TEXT)

JSON fields (labels, participants, etc.) are text — use \`json_extract(labels, '$[0]')\` or \`labels LIKE '%"IMPORTANT"%'\`. Dates are ISO UTC strings; use SQLite's date() / datetime() functions (e.g. \`date(date) >= date('now', '-7 days')\`).

Useful idioms:
- Count per sender, last 30 days: \`SELECT from_addr, COUNT(*) n FROM messages WHERE date >= date('now','-30 days') GROUP BY from_addr ORDER BY n DESC LIMIT 10\`
- Threads with attachments: \`SELECT ts.thread_id, ts.subject FROM thread_summary ts JOIN attachments a ON a.message_id IN (SELECT id FROM messages WHERE thread_id=ts.thread_id) GROUP BY ts.thread_id\`
- Unread from vendor: \`SELECT * FROM messages WHERE from_addr LIKE '%@vendor.com%' AND labels LIKE '%"UNREAD"%'\`

ALWAYS include a LIMIT unless aggregating, otherwise results get truncated and you lose precision. Do NOT SELECT from embeddings, query_cache, messages_fts, or attachments_fts — they're huge or virtual.

Results come back as {columns: string[], rows: unknown[][], row_count, truncated}. The thread_id column gives you a cite_ref (first 8 chars) for citations.`,
    inputSchema: z.object({
      queries: z
        .array(z.string().min(1).max(5000))
        .min(1)
        .max(MAX_BATCH)
        .describe("One or more SQL SELECT queries to run in parallel."),
    }),
    execute: async ({ queries }) =>
      safely("sql_query", () =>
        runBatch(
          "sql_query",
          queries,
          (q) => ({ query: q }),
          async (q) => {
            const r = await runSqlBackend(q);
            if (r.row_count === 0) {
              return {
                ...r,
                quality_note:
                  "Zero rows returned. The WHERE clause may be too narrow, or the data may be stored differently than you expected. Try loosening filters, LIKE-matching a substring, or inspecting the schema with a different query first.",
              };
            }
            return r;
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
            const note =
              detail.messages.length === 0
                ? `Thread ${t} has no messages — it may not exist or be empty. Don't invent content.`
                : null;
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
              ...(note ? { quality_note: note } : {}),
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
          async (id) => {
            const att = await getAttachmentTextBackend(id);
            if (!att.extracted_text || att.extracted_text.length < 20) {
              return {
                ...att,
                quality_note: `Attachment ${id} (${att.filename}) has ${att.extracted_text?.length ?? 0} chars of extracted text — likely an image, scan, or a format we can't extract. Try get_attachment_image or get_attachment_pdf, or tell the user this attachment isn't searchable.`,
              };
            }
            return att;
          },
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
