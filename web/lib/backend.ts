import { pythonApiUrl } from "./config";

export type ThreadMatch = {
  message_id: string;
  score: number;
  from_addr: string;
  date: string;
  snippet: string;
  match_type: string;
  attachment_filename: string | null;
  summary?: string;
};

export type SearchThread = {
  thread_id: string;
  score: number;
  similarity: number;
  subject: string;
  participants: string[];
  message_count: number;
  date_first: string;
  date_last: string;
  user_replied: boolean;
  matches: ThreadMatch[];
};

export type QueryThread = {
  thread_id: string;
  subject: string;
  participants: string[];
  message_count: number;
  date_first: string;
  date_last: string;
  snippet: string;
};

export const searchEmailsBackend = async (args: {
  query: string;
  top_k?: number;
  date_from?: string;
  date_to?: string;
}): Promise<SearchThread[]> => {
  const url = new URL(`${pythonApiUrl()}/api/search`);
  url.searchParams.set("q", args.query);
  url.searchParams.set("k", String(args.top_k ?? 10));
  if (args.date_from) url.searchParams.set("date_from", args.date_from);
  if (args.date_to) url.searchParams.set("date_to", args.date_to);
  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error(`search_emails backend failed: ${res.status}`);
  }
  const data = await res.json();
  return data.results ?? [];
};

export type QueryArgs = {
  sender?: string;
  subject_contains?: string;
  date_from?: string;
  date_to?: string;
  label?: string;
  has_attachment?: boolean;
  order_by?: "date_desc" | "date_asc";
  limit?: number;
};

export const queryEmailsBackend = async (args: QueryArgs): Promise<QueryThread[]> => {
  const url = new URL(`${pythonApiUrl()}/api/query`);
  for (const [key, value] of Object.entries(args)) {
    if (value === undefined || value === null || value === "") continue;
    url.searchParams.set(key, String(value));
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error(`query_emails backend failed: ${res.status}`);
  }
  const data = await res.json();
  return data.results ?? [];
};

export type AttachmentMeta = {
  id: number;
  filename: string;
  mime_type: string;
  size_bytes: number;
};

export type ThreadMessage = {
  id: string;
  from_addr: string;
  to_addr: string;
  subject: string;
  body_text: string;
  date: string;
  labels: string[];
  attachments: AttachmentMeta[];
};

export type ThreadDetail = {
  thread_id: string;
  messages: ThreadMessage[];
};

export const getThreadBackend = async (threadId: string): Promise<ThreadDetail> => {
  const res = await fetch(`${pythonApiUrl()}/api/thread/${encodeURIComponent(threadId)}`);
  if (!res.ok) {
    throw new Error(`get_thread backend failed: ${res.status}`);
  }
  return (await res.json()) as ThreadDetail;
};

export type ThreadLookupHit = { thread_id: string; subject: string };
export type ThreadLookupResult =
  | { ok: true; hit: ThreadLookupHit }
  | { ok: false; status: number; error: string; candidates?: ThreadLookupHit[] };

export const lookupThreadByCiteRef = async (citeRef: string): Promise<ThreadLookupResult> => {
  const url = new URL(`${pythonApiUrl()}/api/thread_lookup`);
  url.searchParams.set("cite_ref", citeRef);
  const res = await fetch(url.toString());
  if (res.ok) {
    const hit = (await res.json()) as ThreadLookupHit;
    return { ok: true, hit };
  }
  let body: { error?: string; candidates?: ThreadLookupHit[] } = {};
  try {
    body = (await res.json()) as typeof body;
  } catch {
    body = { error: `lookup failed (${res.status})` };
  }
  return {
    ok: false,
    status: res.status,
    error: body.error ?? `lookup failed (${res.status})`,
    candidates: body.candidates,
  };
};

export type SqlResult = {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
};

export const runSqlBackend = async (query: string): Promise<SqlResult> => {
  const res = await fetch(`${pythonApiUrl()}/api/sql`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    let msg = `sql_query backend ${res.status}`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body.error) msg = body.error;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return (await res.json()) as SqlResult;
};

export type AttachmentText = {
  attachment_id: number;
  filename: string;
  mime_type: string;
  extracted_text: string;
};

export type MessageDetail = {
  id: string;
  thread_id: string;
  from_addr: string;
  to_addr: string;
  subject: string;
  body_text: string;
  date: string;
  labels: string[];
  attachments: AttachmentMeta[];
};

export const getMessageBackend = async (messageId: string): Promise<MessageDetail> => {
  const res = await fetch(`${pythonApiUrl()}/api/message/${encodeURIComponent(messageId)}`);
  if (!res.ok) {
    throw new Error(`get_message backend failed: ${res.status}`);
  }
  return (await res.json()) as MessageDetail;
};

export const getAttachmentTextBackend = async (attachmentId: number): Promise<AttachmentText> => {
  const res = await fetch(`${pythonApiUrl()}/api/attachment/${attachmentId}/text`);
  if (!res.ok) {
    throw new Error(`get_attachment_text backend failed: ${res.status}`);
  }
  return (await res.json()) as AttachmentText;
};

export type AttachmentBytes = {
  filename: string;
  mimeType: string;
  base64: string;
  sizeBytes: number;
};

export type CorpusStatus = {
  messages: number;
  embeddings: number;
  date_oldest: string | null;
  date_newest: string | null;
  total_cost_usd: number;
  budget_remaining_usd: number;
  running_job: { stage?: string } | null;
};

export const getCorpusStatusBackend = async (): Promise<CorpusStatus> => {
  const res = await fetch(`${pythonApiUrl()}/api/status`);
  if (!res.ok) throw new Error(`status backend failed: ${res.status}`);
  return (await res.json()) as CorpusStatus;
};

export const getAttachmentBytesBackend = async (attachmentId: number): Promise<AttachmentBytes> => {
  const res = await fetch(`${pythonApiUrl()}/api/attachment/${attachmentId}`);
  if (!res.ok) {
    throw new Error(`get_attachment backend failed: ${res.status}`);
  }
  const mimeType = res.headers.get("content-type") || "application/octet-stream";
  const filename = parseFilenameFromContentDisposition(res.headers.get("content-disposition"));
  const buf = Buffer.from(await res.arrayBuffer());
  return {
    filename: filename ?? `attachment-${attachmentId}`,
    mimeType,
    base64: buf.toString("base64"),
    sizeBytes: buf.byteLength,
  };
};

const parseFilenameFromContentDisposition = (header: string | null): string | null => {
  if (!header) return null;
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header);
  return match?.[1] ?? null;
};
