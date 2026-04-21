"use client";

import { useEffect, useState, useSyncExternalStore } from "react";

import { cleanSender } from "@/lib/sender";
import { fetchThread, getCachedThread, subscribeThread } from "@/lib/threadCache";

export type ThreadHint = {
  thread_id: string;
  subject?: string;
  participants?: string[];
};

type Props = {
  threadId: string;
  hints: ThreadHint[];
  onOpen: (threadId: string) => void;
};

const findInHints = (threadId: string, hints: ThreadHint[]) => {
  const hit = hints.find((t) => t.thread_id === threadId);
  if (!hit) return null;
  return {
    subject: hit.subject ?? "",
    sender: Array.isArray(hit.participants) ? hit.participants[0] ?? "" : "",
  };
};

const useCachedThread = (threadId: string) =>
  useSyncExternalStore(
    (cb) => subscribeThread(threadId, cb),
    () => getCachedThread(threadId),
    () => undefined,
  );

export const CitationChip = ({ threadId, hints, onOpen }: Props) => {
  const cached = useCachedThread(threadId);
  const [didFetch, setDidFetch] = useState(false);

  useEffect(() => {
    if (findInHints(threadId, hints)) return;
    if (didFetch) return;
    if (cached && cached !== "error") return;
    setDidFetch(true);
    void fetchThread(threadId);
  }, [threadId, hints, cached, didFetch]);

  const fromTool = findInHints(threadId, hints);
  const fromCache =
    cached && cached !== "loading" && cached !== "error" && cached.messages.length > 0
      ? {
          subject: cached.messages[0].subject,
          sender: cached.messages[0].from_addr,
        }
      : null;
  const meta = fromTool ?? fromCache;
  const isLoading = !meta && cached === "loading";

  const subject = meta?.subject ?? (isLoading ? "Loading…" : threadId.slice(0, 10));
  const sender = cleanSender(meta?.sender ?? "");

  return (
    <button
      type="button"
      onClick={() => onOpen(threadId)}
      className="inline-flex items-center gap-1.5 rounded-md bg-blue-50 hover:bg-blue-100 transition-colors px-2 py-0.5 text-xs text-blue-800 align-baseline border border-blue-100 max-w-[28ch]"
      title={subject}
    >
      <svg className="w-3 h-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
      <span className="truncate font-medium">{subject}</span>
      {sender && <span className="truncate text-blue-600/70 font-normal">· {sender}</span>}
    </button>
  );
};
