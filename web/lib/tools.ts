import { dynamicTool, tool } from "ai";
import { z } from "zod";

import {
  getAttachmentBytesBackend,
  getAttachmentRenderedPagesBackend,
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
// model can call get_attachment with 10 inline_pdf requests × 15 MB
// each and we ship 150 MB of base64 to Gemini in one request — which
// fails noisily if it doesn't OOM Node first.
// NOTE: counted in BASE64 CHARS below (not decoded bytes), so this is
// roughly a 25 MB wire-format budget. Keep in sync with the Python
// `_INLINE_BYTES_CAP` in src/gmail_search/server.py (its unit is
// decoded bytes so effective thresholds differ by ~1.33×).
const MAX_BATCH_INLINE_BYTES = 25 * 1024 * 1024;
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

// Per-message body cap for get_thread. A single chatty newsletter thread
// with 30 quoted-reply messages × 50KB bodies × 10 threads in one batch
// can ship >15MB back to the model, blowing Gemini's 1M-token input cap
// on the next agent step. 20K chars (~5K tokens) preserves the most
// recent reply in full for ~99% of real email threads; the model sees
// a `body_text_truncated: true` flag + the original length when we cut.
const THREAD_BODY_CHAR_CAP = 20000;

// Per-cell cap for sql_query results. sql_query caps at 500 rows but
// not cell size — `SELECT body_text FROM messages LIMIT 500` can ship
// 10+ MB of text. 8K chars per string cell covers normal columns
// (subject, from_addr, etc.) with huge headroom and truncates long
// body_text cleanly.
const SQL_CELL_CHAR_CAP = 8000;

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

// Cap a per-message body for get_thread. Returns the clipped text plus
// metadata the model can use to decide whether to dig deeper.
const capThreadBody = (
  body: string,
): { body_text: string; original_chars: number; body_text_truncated?: true } => {
  const original = body ?? "";
  if (original.length <= THREAD_BODY_CHAR_CAP) {
    return { body_text: original, original_chars: original.length };
  }
  return {
    body_text: original.slice(0, THREAD_BODY_CHAR_CAP),
    original_chars: original.length,
    body_text_truncated: true,
  };
};

// Clamp a single sql_query cell. Non-strings pass through. Long strings
// are clipped with a sentinel so the model knows content was dropped.
const capSqlCell = (cell: unknown): { value: unknown; truncated: boolean } => {
  if (typeof cell !== "string" || cell.length <= SQL_CELL_CHAR_CAP) {
    return { value: cell, truncated: false };
  }
  const kept = cell.slice(0, SQL_CELL_CHAR_CAP);
  return {
    value: `${kept}… [truncated: original ${cell.length} chars]`,
    truncated: true,
  };
};

// Walk every cell in a SqlResult and clamp long strings. Adds a
// `cells_truncated` count so the model sees the scope of the clip.
const capSqlResult = <T extends { rows: unknown[][] }>(
  result: T,
): T & { cells_truncated?: number } => {
  let truncatedCount = 0;
  const cappedRows = result.rows.map((row) =>
    row.map((cell) => {
      const { value, truncated } = capSqlCell(cell);
      if (truncated) truncatedCount += 1;
      return value;
    }),
  );
  if (truncatedCount === 0) return result;
  return { ...result, rows: cappedRows, cells_truncated: truncatedCount };
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

// Strip RTL/LRO/zero-width control chars and other invisible bidi tricks an
// attacker might embed in an email to hide injected instructions from the
// human while still feeding them to the model.
const sanitizeBodyExcerpt = (text: string): string =>
  text.replace(/[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]/g, "");

const fetchTopBody = async (messageId: string | null): Promise<string | null> => {
  if (!messageId) return null;
  try {
    const msg = await getMessageBackend(messageId);
    return sanitizeBodyExcerpt((msg.body_text ?? "").slice(0, ENRICHED_BODY_CHARS));
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
    return sanitizeBodyExcerpt((latest.body_text ?? "").slice(0, QUERY_ENRICHED_BODY_CHARS));
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
  date_from: z
    .string()
    .optional()
    .describe(
      "ISO date YYYY-MM-DD (inclusive). Restricts results to threads with matching messages on or after this date. Relevance ranking still applies within the window.",
    ),
  date_to: z
    .string()
    .optional()
    .describe("ISO date YYYY-MM-DD (inclusive) upper bound."),
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
      "Semantic + keyword hybrid search over the user's Gmail archive. BATCH: pass one OR many queries in `searches` and they run in parallel. OPTIONAL DATE FILTER: each search can pass date_from/date_to (ISO YYYY-MM-DD) to restrict results to a time window while KEEPING relevance ranking inside it — use this for mixed questions like 'construction emails last week' instead of running query_emails and losing relevance. Every thread carries a `summary` (1-3 sentences, local-model) and the top 5 threads additionally carry a `body_excerpt` (up to 4000 chars of the matching message). You usually do NOT need get_thread — only call it if the body_excerpt is too short or you need multiple messages from the same thread.",
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
            const raw = await searchEmailsBackend({
              query: s.query,
              top_k: s.top_k,
              date_from: s.date_from,
              date_to: s.date_to,
            });
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

The full schema (tables and columns) is in the <sql_schema> section of the system prompt — that is the AUTHORITATIVE source. Do not guess column names from memory; read <sql_schema> first.

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
            const r = capSqlResult(await runSqlBackend(q));
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
                ...capThreadBody(m.body_text),
                attachments: m.attachments,
              })),
              ...(note ? { quality_note: note } : {}),
            };
          },
        ),
      ),
  }),

  // Single attachment executor. The model reads the manifest on each
  // attachment row (text_chars, can_inline_pdf, can_inline_image,
  // can_render_pages, suggested_as) from get_thread / get_message /
  // query_emails results and tells us what representation it wants for
  // each one. Server is a dumb executor — no "auto" mode.
  get_attachment: dynamicTool({
    description:
      "Fetch one or more attachments in the representation the model picks. Input is a batched array of `{attachment_id, as, pages?}` requests. Choices:\n" +
      "  - \"text\": already-extracted text (PDF/docx/csv/txt/calendar). Cheapest. Use when `text_chars` on the manifest is non-trivial (>= a few hundred).\n" +
      "  - \"inline_pdf\": raw PDF bytes inlined. Use when `can_inline_pdf` is true AND layout/scans matter or text is empty. Reads natively; no OCR step needed by the model.\n" +
      "  - \"inline_image\": raw image bytes inlined. Use when `can_inline_image` is true. Always pick this for image/* attachments when visual content matters.\n" +
      "  - \"rendered_pages\": server rasterizes PDF pages to PNG and inlines them as images. Fallback for PDFs where `inline_pdf` fails or when you want SPECIFIC pages (pass `pages: [1,2,3]`). Useful for scans.\n" +
      "BATCH: send every attachment you need in one call; per-item mode lets you mix representations in a single request (e.g. one PDF as text, another as inline_pdf). Max 6 requests per call.",
    // Discriminated on `as` so each representation gets a proper schema
    // (only `rendered_pages` carries `pages`). Without this the model
    // could pass `{as: "text", pages: [1,2,3]}` and we'd silently
    // ignore it — bug-hiding, better to reject at the schema level.
    inputSchema: z.object({
      requests: z
        .array(
          z.discriminatedUnion("as", [
            z.object({
              attachment_id: z.number().int(),
              as: z.literal("text"),
            }),
            z.object({
              attachment_id: z.number().int(),
              as: z.literal("inline_pdf"),
            }),
            z.object({
              attachment_id: z.number().int(),
              as: z.literal("inline_image"),
            }),
            z.object({
              attachment_id: z.number().int(),
              as: z.literal("rendered_pages"),
              pages: z
                .array(z.number().int().min(1))
                .optional()
                .describe("1-based page numbers. Omit for first N pages."),
            }),
          ]),
        )
        .min(1)
        .max(6)
        .describe("One directive per attachment."),
    }),
    execute: async (input) =>
      safely("get_attachment", async () => {
        const { requests } = input as {
          requests: Array<
            | { attachment_id: number; as: "text" }
            | { attachment_id: number; as: "inline_pdf" }
            | { attachment_id: number; as: "inline_image" }
            | { attachment_id: number; as: "rendered_pages"; pages?: number[] }
          >;
        };
        // Byte accounting is in base64 CHARACTERS because that's what we
        // actually ship to the model. Decoded-byte counting undercounts
        // by 33% (base64 expands 4/3). MAX_BATCH_INLINE_BYTES is thus a
        // cap on aggregate base64 length — comparing `sizeBytes` (raw
        // binary) against it undercounted; comparing `b64.length * 0.75`
        // undercounted twice. Count `.base64.length` everywhere.
        let bytesUsed = 0;
        const results = await Promise.all(
          requests.map(async (req) => {
            try {
              if (req.as === "text") {
                const t = await getAttachmentTextBackend(req.attachment_id);
                const text = sanitizeBodyExcerpt(t.extracted_text ?? "");
                return {
                  attachment_id: req.attachment_id,
                  as: "text" as const,
                  filename: t.filename,
                  mime_type: t.mime_type,
                  text,
                  text_chars: text.length,
                  quality_note:
                    text.length < 20
                      ? `Only ${text.length} chars extracted. If more content is needed, retry this attachment as "inline_pdf" or "rendered_pages".`
                      : undefined,
                };
              }
              if (req.as === "inline_pdf" || req.as === "inline_image") {
                const att = await getAttachmentBytesBackend(req.attachment_id);
                const expectPdf = req.as === "inline_pdf";
                const mimeOk = expectPdf
                  ? att.mimeType === "application/pdf"
                  : att.mimeType.startsWith("image/");
                if (!mimeOk) {
                  return {
                    attachment_id: req.attachment_id,
                    as: req.as,
                    error: `mime mismatch: ${att.mimeType}, expected ${expectPdf ? "application/pdf" : "image/*"}`,
                  };
                }
                const b64Bytes = att.base64.length;
                if (b64Bytes > MAX_INLINE_BYTES) {
                  return {
                    attachment_id: req.attachment_id,
                    as: req.as,
                    error: `Too big to inline (~${(b64Bytes / 1024 / 1024).toFixed(1)}MB base64 > ${MAX_INLINE_BYTES / 1024 / 1024}MB). Retry as "rendered_pages" or "text".`,
                  };
                }
                if (bytesUsed + b64Bytes > MAX_BATCH_INLINE_BYTES) {
                  return {
                    attachment_id: req.attachment_id,
                    as: req.as,
                    error: `Batch byte cap hit — call again with fewer inline items.`,
                  };
                }
                bytesUsed += b64Bytes;
                return {
                  attachment_id: req.attachment_id,
                  as: req.as,
                  filename: att.filename,
                  mime_type: att.mimeType,
                  size_bytes: att.sizeBytes,
                  base64: att.base64,
                };
              }
              if (req.as === "rendered_pages") {
                const r = await getAttachmentRenderedPagesBackend(req.attachment_id, req.pages);
                // Same cap as inline_* paths, counted in base64 chars.
                const batchBytes = r.pages.reduce((n, p) => n + p.base64.length, 0);
                if (bytesUsed + batchBytes > MAX_BATCH_INLINE_BYTES) {
                  return {
                    attachment_id: req.attachment_id,
                    as: req.as,
                    error: `Batch byte cap hit (${r.pages.length} pages would overflow). Retry with fewer pages.`,
                  };
                }
                bytesUsed += batchBytes;
                return {
                  attachment_id: req.attachment_id,
                  as: req.as,
                  total_pages: r.total_pages,
                  pages: r.pages, // [{page, base64, mime_type}]
                };
              }
              return { attachment_id: (req as { attachment_id: number }).attachment_id, as: "unknown", error: `unknown as` };
            } catch (e) {
              return {
                attachment_id: req.attachment_id,
                as: req.as,
                error: e instanceof Error ? e.message : String(e),
              };
            }
          }),
        );
        return { results };
      }),
    // Flatten the batched result into the interleaved content parts the
    // model actually consumes: a text header for each item, then the
    // binary where applicable. Keeps one result in one logical spot
    // (no "all headers, then all bytes" confusion).
    toModelOutput: ({ output }) => {
      const o = output as { results?: Array<Record<string, unknown>> } | { error?: string };
      if (!o || typeof o !== "object" || "error" in o) {
        return { type: "json", value: { error: String((o as { error?: string })?.error ?? "unknown") } };
      }
      const results = (o as { results: Array<Record<string, unknown>> }).results;
      type Part =
        | { type: "text"; text: string }
        | { type: "image-data"; data: string; mediaType: string }
        | { type: "file-data"; data: string; mediaType: string; filename: string };
      const parts: Part[] = [];
      for (const r of results) {
        const id = r.attachment_id;
        const as = String(r.as ?? "");
        if (r.error) {
          parts.push({ type: "text", text: `attachment ${id} (${as}) error: ${String(r.error)}` });
          continue;
        }
        if (as === "text") {
          const text = String(r.text ?? "");
          const note = r.quality_note ? ` [${String(r.quality_note)}]` : "";
          parts.push({
            type: "text",
            text: `--- attachment ${id} (${String(r.filename ?? "")}, text, ${r.text_chars} chars)${note} ---\n${text}`,
          });
          continue;
        }
        if (as === "inline_pdf") {
          if (!r.base64) {
            parts.push({ type: "text", text: `attachment ${id} (inline_pdf) no data` });
            continue;
          }
          parts.push({ type: "text", text: `--- attachment ${id} (${String(r.filename ?? "")}, inline_pdf) ---` });
          parts.push({
            type: "file-data",
            data: String(r.base64),
            mediaType: String(r.mime_type ?? "application/pdf"),
            filename: String(r.filename ?? `attachment-${id}.pdf`),
          });
          continue;
        }
        if (as === "inline_image") {
          if (!r.base64) {
            parts.push({ type: "text", text: `attachment ${id} (inline_image) no data` });
            continue;
          }
          parts.push({ type: "text", text: `--- attachment ${id} (${String(r.filename ?? "")}, inline_image) ---` });
          parts.push({
            type: "image-data",
            data: String(r.base64),
            mediaType: String(r.mime_type ?? "image/png"),
          });
          continue;
        }
        if (as === "rendered_pages") {
          const pages = (r.pages as Array<{ page: number; base64: string; mime_type: string }>) ?? [];
          parts.push({
            type: "text",
            text: `--- attachment ${id} (rendered_pages, ${pages.length}/${r.total_pages} pages) ---`,
          });
          for (const p of pages) {
            parts.push({ type: "text", text: `[page ${p.page}]` });
            parts.push({ type: "image-data", data: p.base64, mediaType: p.mime_type });
          }
          continue;
        }
        parts.push({ type: "text", text: `attachment ${id} (${as}) unknown representation` });
      }
      return { type: "content", value: parts };
    },
  }),
});

export type AppTools = ReturnType<typeof buildTools>;
