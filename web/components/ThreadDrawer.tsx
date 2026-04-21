"use client";

import { useEffect, useState } from "react";

import { EmailBody } from "@/components/EmailBody";
import { fetchThread } from "@/lib/threadCache";
import type { ThreadDetail, ThreadMessage } from "@/lib/backend";

import { Drawer } from "./Drawer";

type Props = {
  threadId: string | null;
  onClose: () => void;
  pythonBaseUrl: string;
};

const formatDate = (iso: string) => {
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return iso;
  }
};

const cleanSender = (raw: string): string => {
  const angle = raw.match(/^([^<]+)</);
  return (angle?.[1] ?? raw).replace(/"/g, "").trim();
};

const senderInitial = (raw: string): string => {
  const c = cleanSender(raw).charAt(0).toUpperCase();
  return c || "?";
};

const senderColor = (raw: string): string => {
  const palette = ["bg-rose-500", "bg-amber-500", "bg-emerald-500", "bg-sky-500", "bg-violet-500", "bg-fuchsia-500", "bg-teal-500"];
  let hash = 0;
  for (let i = 0; i < raw.length; i++) hash = (hash * 31 + raw.charCodeAt(i)) >>> 0;
  return palette[hash % palette.length];
};

const Avatar = ({ from }: { from: string }) => (
  <div
    className={`w-9 h-9 rounded-full text-white flex items-center justify-center text-sm font-semibold shrink-0 ${senderColor(from)}`}
  >
    {senderInitial(from)}
  </div>
);

const AttachmentPills = ({ attachments, pythonBaseUrl }: { attachments: ThreadMessage["attachments"]; pythonBaseUrl: string }) => {
  if (attachments.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {attachments.map((a) => (
        <a
          key={a.id}
          href={`${pythonBaseUrl}/api/attachment/${a.id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 rounded border border-neutral-200 bg-white px-2 py-1 text-xs text-neutral-700 hover:bg-neutral-50"
          title={`${a.mime_type} · ${(a.size_bytes / 1024).toFixed(0)}KB`}
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
          </svg>
          <span className="truncate max-w-[28ch]">{a.filename}</span>
        </a>
      ))}
    </div>
  );
};

const MessageCard = ({ msg, pythonBaseUrl }: { msg: ThreadMessage; pythonBaseUrl: string }) => (
  <div className="flex gap-3 py-4 border-b border-neutral-100 last:border-0">
    <Avatar from={msg.from_addr} />
    <div className="flex-1 min-w-0">
      <div className="flex items-baseline justify-between gap-2">
        <div className="font-medium text-sm text-neutral-900 truncate">{cleanSender(msg.from_addr)}</div>
        <div className="text-xs text-neutral-500 shrink-0">{formatDate(msg.date)}</div>
      </div>
      {msg.to_addr && (
        <div className="text-xs text-neutral-500 truncate">to {msg.to_addr}</div>
      )}
      <div className="mt-2">
        <EmailBody textBody={msg.body_text} htmlBody={msg.body_html} />
      </div>
      <AttachmentPills attachments={msg.attachments} pythonBaseUrl={pythonBaseUrl} />
    </div>
  </div>
);

export const ThreadDrawer = ({ threadId, onClose, pythonBaseUrl }: Props) => {
  const [detail, setDetail] = useState<ThreadDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!threadId) {
      setDetail(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setDetail(null);
    setError(null);
    void fetchThread(threadId).then((data) => {
      if (cancelled) return;
      if (data) setDetail(data);
      else setError("Could not load thread.");
    });
    return () => {
      cancelled = true;
    };
  }, [threadId]);

  const open = threadId !== null;
  const subject = detail?.messages[0]?.subject ?? (open ? "Loading…" : "");
  const subtitle =
    detail && `${detail.messages.length} message${detail.messages.length === 1 ? "" : "s"}`;

  return (
    <Drawer open={open} onClose={onClose} title={subject} subtitle={subtitle}>
      <div className="px-5">
        {error && <div className="py-8 text-sm text-red-600">{error}</div>}
        {!error && !detail && open && (
          <div className="py-8 text-sm text-neutral-500">Loading thread…</div>
        )}
        {detail &&
          detail.messages.map((m) => <MessageCard key={m.id} msg={m} pythonBaseUrl={pythonBaseUrl} />)}
      </div>
    </Drawer>
  );
};
