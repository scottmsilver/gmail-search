"use client";

import type { SearchThread } from "@/lib/backend";
import { formatSmartDate } from "@/lib/datetime";

const cleanSender = (raw: string): string => {
  const angle = raw.match(/^([^<]+)</);
  return (angle?.[1] ?? raw).replace(/"/g, "").trim();
};

const senderInitial = (raw: string): string =>
  cleanSender(raw).charAt(0).toUpperCase() || "?";

const PALETTE = [
  "bg-rose-500",
  "bg-amber-500",
  "bg-emerald-500",
  "bg-sky-500",
  "bg-violet-500",
  "bg-fuchsia-500",
  "bg-teal-500",
];

const senderColor = (raw: string): string => {
  let h = 0;
  for (let i = 0; i < raw.length; i++) h = (h * 31 + raw.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
};

const cleanSnippet = (s: string): string =>
  s.replace(/^From:.*?\| /g, "").replace(/\r?\n/g, " ").replace(/\s+/g, " ").trim();

const shortPeople = (participants: string[]): string => {
  const names = participants.map(cleanSender).filter(Boolean);
  if (names.length <= 3) return names.join(", ");
  return `${names[0]}, ${names[1]}, +${names.length - 2}`;
};

const Avatar = ({ from }: { from: string }) => (
  <div
    className={`w-8 h-8 rounded-full text-white flex items-center justify-center text-xs font-semibold shrink-0 ${senderColor(from)}`}
  >
    {senderInitial(from)}
  </div>
);

const PaperclipIcon = () => (
  <svg className="w-3.5 h-3.5 inline-block opacity-60 mr-1 -mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
  </svg>
);

const ReplyIcon = () => (
  <svg className="w-3 h-3 inline-block text-emerald-600 mr-1" viewBox="0 0 24 24" fill="currentColor">
    <path d="M10 9V5l-7 7 7 7v-4.1c5 0 8.5 1.6 11 5.1-1-5-4-10-11-11z" />
  </svg>
);

type Props = {
  thread: SearchThread;
  onOpen: (threadId: string) => void;
};

export const ResultRow = ({ thread, onOpen }: Props) => {
  const top = thread.matches[0];
  const snippet = top ? cleanSnippet(top.snippet).slice(0, 180) : "";
  const hasAttachment = thread.matches.some((m) => m.attachment_filename);
  const senderForAvatar = top?.from_addr ?? thread.participants[0] ?? "?";

  return (
    <button
      type="button"
      onClick={() => onOpen(thread.thread_id)}
      className="group flex w-full items-start gap-3 border-b px-4 py-3 text-left transition-colors hover:bg-accent/50"
    >
      <Avatar from={senderForAvatar} />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-3">
          <div className="truncate text-sm font-medium text-foreground">
            {thread.user_replied && <ReplyIcon />}
            {shortPeople(thread.participants)}
            {thread.message_count > 1 && (
              <span className="ml-1.5 text-xs font-normal text-muted-foreground">{thread.message_count}</span>
            )}
          </div>
          <div className="shrink-0 text-xs text-muted-foreground">{formatSmartDate(thread.date_last)}</div>
        </div>
        <div className="mt-0.5 truncate text-sm text-muted-foreground">
          {hasAttachment && <PaperclipIcon />}
          <span className="font-medium text-foreground">{thread.subject}</span>
          {snippet && (
            <>
              <span className="mx-1.5 text-muted-foreground">—</span>
              <span className="text-muted-foreground">{snippet}</span>
            </>
          )}
        </div>
      </div>
    </button>
  );
};
