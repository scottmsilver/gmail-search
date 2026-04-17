import { dynamicTool, tool } from "ai";
import { z } from "zod";

import {
  getAttachmentBytesBackend,
  getAttachmentTextBackend,
  getThreadBackend,
  queryEmailsBackend,
  searchEmailsBackend,
} from "./backend";

const MAX_INLINE_BYTES = 15 * 1024 * 1024;

const formatSearchOutput = (raw: Awaited<ReturnType<typeof searchEmailsBackend>>) =>
  raw.map((t) => {
    const top = t.matches[0];
    return {
      thread_id: t.thread_id,
      subject: t.subject,
      participants: t.participants,
      message_count: t.message_count,
      date_last: t.date_last,
      from_addr: top?.from_addr ?? "",
      snippet: top?.snippet ?? "",
      score: Number(t.score.toFixed(3)),
    };
  });

const formatQueryOutput = (raw: Awaited<ReturnType<typeof queryEmailsBackend>>) =>
  raw.map((t) => ({
    thread_id: t.thread_id,
    subject: t.subject,
    participants: t.participants,
    message_count: t.message_count,
    date_last: t.date_last,
    snippet: t.snippet,
  }));

export const buildTools = () => ({
  search_emails: tool({
    description:
      "Semantic + keyword hybrid search over the user's Gmail archive. Use for any question where relevance to a topic matters. Returns threads ranked by relevance with short snippets only — call get_thread to read full content.",
    inputSchema: z.object({
      query: z.string().describe("Natural language query."),
      top_k: z.number().int().min(1).max(20).optional().describe("Threads to return (default 10)."),
    }),
    execute: async ({ query, top_k }) => {
      const raw = await searchEmailsBackend({ query, top_k });
      return { threads: formatSearchOutput(raw) };
    },
  }),

  query_emails: tool({
    description:
      "Structured filter over Gmail by metadata. Use for deterministic questions like 'newest from Alice', 'emails from March from bank.com'. Returns short snippets only — call get_thread for full content.",
    inputSchema: z.object({
      sender: z.string().optional().describe("Substring match on from address."),
      subject_contains: z.string().optional(),
      date_from: z.string().optional().describe("ISO date YYYY-MM-DD, inclusive."),
      date_to: z.string().optional().describe("ISO date YYYY-MM-DD, inclusive."),
      label: z.string().optional().describe("Gmail label (INBOX, IMPORTANT, SENT, UNREAD)."),
      has_attachment: z.boolean().optional(),
      order_by: z.enum(["date_desc", "date_asc"]).optional(),
      limit: z.number().int().min(1).max(100).optional().describe("Default 20, max 100."),
    }),
    execute: async (args) => {
      const raw = await queryEmailsBackend(args);
      return { threads: formatQueryOutput(raw) };
    },
  }),

  get_thread: tool({
    description:
      "Fetch every message in a thread with full body text and an attachment list (id, filename, mime_type, size). Use this after search/query when snippets aren't enough.",
    inputSchema: z.object({
      thread_id: z.string().describe("Thread ID from search/query results."),
    }),
    execute: async ({ thread_id }) => {
      const detail = await getThreadBackend(thread_id);
      return {
        thread_id: detail.thread_id,
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
  }),

  get_attachment_text: tool({
    description:
      "Return text already extracted from an attachment (PDFs, docx, csv, calendar invites, etc.). Cheap — use first when you only need the words.",
    inputSchema: z.object({
      attachment_id: z.number().int().describe("Attachment ID from get_thread."),
    }),
    execute: async ({ attachment_id }) => {
      const att = await getAttachmentTextBackend(attachment_id);
      return att;
    },
  }),

  get_attachment_image: dynamicTool({
    description:
      "Inline an image attachment directly so you can see it. Only use for image/* attachments and only when visual content matters.",
    inputSchema: z.object({
      attachment_id: z.number().int(),
    }),
    execute: async (input) => {
      const { attachment_id } = input as { attachment_id: number };
      const att = await getAttachmentBytesBackend(attachment_id);
      if (!att.mimeType.startsWith("image/")) {
        return { error: `Attachment is ${att.mimeType}, not image/*. Try get_attachment_pdf or get_attachment_text.` };
      }
      if (att.sizeBytes > MAX_INLINE_BYTES) {
        return { error: `Image is ${(att.sizeBytes / 1024 / 1024).toFixed(1)}MB — too large to inline.` };
      }
      return {
        filename: att.filename,
        mimeType: att.mimeType,
        sizeBytes: att.sizeBytes,
        base64: att.base64,
      };
    },
    toModelOutput: ({ output }) => {
      const o = output as Record<string, unknown>;
      if (!o || typeof o !== "object" || "error" in o) {
        return { type: "json", value: { error: String(o?.error ?? "unknown") } };
      }
      return {
        type: "content",
        value: [
          { type: "text", text: `Inlined image: ${String(o.filename)}` },
          { type: "image-data", data: String(o.base64), mediaType: String(o.mimeType) },
        ],
      };
    },
  }),

  get_attachment_pdf: dynamicTool({
    description:
      "Inline a PDF attachment directly for layout-aware reading (floor plans, signed forms, charts). Prefer get_attachment_text first; only escalate when layout matters.",
    inputSchema: z.object({
      attachment_id: z.number().int(),
    }),
    execute: async (input) => {
      const { attachment_id } = input as { attachment_id: number };
      const att = await getAttachmentBytesBackend(attachment_id);
      if (att.mimeType !== "application/pdf") {
        return { error: `Attachment is ${att.mimeType}, not application/pdf. Try get_attachment_image or get_attachment_text.` };
      }
      if (att.sizeBytes > MAX_INLINE_BYTES) {
        return { error: `PDF is ${(att.sizeBytes / 1024 / 1024).toFixed(1)}MB — too large to inline.` };
      }
      return {
        filename: att.filename,
        mimeType: att.mimeType,
        sizeBytes: att.sizeBytes,
        base64: att.base64,
      };
    },
    toModelOutput: ({ output }) => {
      const o = output as Record<string, unknown>;
      if (!o || typeof o !== "object" || "error" in o) {
        return { type: "json", value: { error: String(o?.error ?? "unknown") } };
      }
      return {
        type: "content",
        value: [
          { type: "text", text: `Inlined PDF: ${String(o.filename)}` },
          {
            type: "file-data",
            data: String(o.base64),
            mediaType: String(o.mimeType),
            filename: String(o.filename),
          },
        ],
      };
    },
  }),
});

export type AppTools = ReturnType<typeof buildTools>;
