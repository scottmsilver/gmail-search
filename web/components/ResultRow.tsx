"use client";

import { useState } from "react";

import { MiddleTruncate } from "@/components/MiddleTruncate";
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

// Embedding chunks are stored with a metadata prefix
// ("From: ... | To: ... | Date: ... | Subject: ... | <body>") so the
// embedder sees context. Strip the whole run from display snippets so
// the UI shows just the message body.
const cleanSnippet = (s: string): string =>
  s
    .replace(/^(?:(?:From|To|Date|Subject):[^|]*\|\s*)+/i, "")
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();

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

// Copy-to-clipboard control. Rendered as <span role="button"> because
// the row's outer element is already a <button>, and HTML forbids
// nested buttons (React 19 now throws a hydration error for this).
// ARIA attributes + keyboard handling keep it accessible.
const CopyButton = ({ text, label }: { text: string; label: string }) => {
  const [copied, setCopied] = useState(false);
  const doCopy = (e: React.SyntheticEvent) => {
    e.stopPropagation();
    e.preventDefault();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  };
  return (
    <span
      role="button"
      tabIndex={0}
      aria-label={`Copy ${label}`}
      onClick={doCopy}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") doCopy(e);
      }}
      title={`Copy ${label}`}
      className="ml-1 inline-flex h-4 w-4 cursor-pointer items-center justify-center rounded text-muted-foreground/60 hover:bg-muted hover:text-foreground"
    >
      {copied ? (
        <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" strokeWidth={3}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" strokeWidth={2}>
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path d="M5 15V5a2 2 0 012-2h10" />
        </svg>
      )}
    </span>
  );
};

type Props = {
  thread: SearchThread;
  onOpen: (threadId: string) => void;
};

// Short form of the stored model key for the debug line. The full
// string is "ciocan/gemma-4-E4B-it-W4A16+v5"; we only need the
// distinguishing tail. Examples:
//   ciocan/gemma-4-E4B-it-W4A16+v5 → "gemma+v5"
//   qwen2.5:7b                     → "qwen2.5"
//   gemma4:latest                  → "gemma4"
const shortModel = (model?: string | null): string => {
  if (!model) return "-";
  const m = model.match(/([^/]+)$/);
  let base = m ? m[1] : model;
  base = base.replace(/^gemma-4-E4B-it-W4A16/, "gemma").replace(/:latest$/, "");
  return base;
};

// "2 min ago" / "3 hr ago" / "4 days ago" — no library, just a
// compact age for the debug footer.
//
// SQLite's CURRENT_TIMESTAMP produces "YYYY-MM-DD HH:MM:SS" with no
// timezone, which JS `Date.parse` interprets as LOCAL time. The
// stored values are actually UTC (SQLite default), so we append "Z"
// before parsing to force UTC. Without this, we end up with negative
// ages when the machine's timezone is west of UTC.
const ageString = (iso?: string | null): string => {
  if (!iso) return "-";
  const normalised = /Z|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : iso.replace(" ", "T") + "Z";
  const t = Date.parse(normalised);
  if (Number.isNaN(t)) return "-";
  const sec = Math.max(0, (Date.now() - t) / 1000);
  if (sec < 60) return `${Math.round(sec)}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
};

export const ResultRow = ({ thread, onOpen }: Props) => {
  const top = thread.matches[0];
  const hasSummary = !!(top?.summary && top.summary.trim());
  // Prefer the LLM-generated summary over the raw matched snippet;
  // fall back to the cleaned snippet when no summary exists.
  const preview = hasSummary ? top!.summary!.trim() : cleanSnippet(top?.snippet ?? "").slice(0, 180);
  const previewSource: "summary" | "snippet" | "empty" = hasSummary
    ? "summary"
    : preview
      ? "snippet"
      : "empty";
  const hasAttachment = thread.matches.some((m) => m.attachment_filename);
  const senderForAvatar = top?.from_addr ?? thread.participants[0] ?? "?";

  // Two-row columnar layout:
  //   Row 1:  avatar · names · subject · date
  //   Row 2:  (avatar spans down)  ·  summary-spanning-names+subject+date
  //
  // The summary gets the full width under sender+subject+date columns
  // rather than being constrained to the subject column alone. Avatar
  // uses `row-span-2` so it doesn't leave an empty cell in row 2.
  return (
    <button
      type="button"
      onClick={() => onOpen(thread.thread_id)}
      className="group grid w-full grid-cols-[auto_200px_1fr_auto] items-start gap-x-3 gap-y-1 border-b px-4 py-2.5 text-left text-sm transition-colors hover:bg-accent/50"
    >
      <div className="row-span-2">
        <Avatar from={senderForAvatar} />
      </div>

      {/* Names (row 1, col 2) */}
      <div className="min-w-0 truncate pt-0.5">
        {thread.user_replied && <ReplyIcon />}
        <span className="font-medium text-foreground">{shortPeople(thread.participants)}</span>
        {thread.message_count > 1 && (
          <span className="ml-1.5 text-xs font-normal text-muted-foreground">{thread.message_count}</span>
        )}
      </div>

      {/* Subject (row 1, col 3) — middle-ellipsised so Gmail's
          "Re: Fwd: Fwd: Re: real subject [EXT] [EXT] [EXT]" threads
          keep the meaningful middle visible instead of getting cut
          off on the right like end-truncate did. */}
      <div className="flex min-w-0 items-center gap-1 font-medium text-foreground">
        {hasAttachment && <PaperclipIcon />}
        <MiddleTruncate text={thread.subject} className="min-w-0 flex-1" />
        {/* Copy Gmail link for this thread — jumps straight to Gmail web. */}
        <CopyButton text={`https://mail.google.com/mail/u/0/#all/${thread.thread_id}`} label="Gmail link" />
      </div>

      {/* Date (row 1, col 4) */}
      <div className="shrink-0 pt-1 text-xs text-muted-foreground">{formatSmartDate(thread.date_last)}</div>

      {/* Summary + debug footer (row 2, spans names+subject+date) */}
      <div className="col-start-2 col-span-3 min-w-0">
        {preview && (
          <div className="whitespace-normal break-words text-muted-foreground">{preview}</div>
        )}
        {/* Debug footer — tells you where each row's data comes from
            so problematic summaries can be traced back to a specific
            prompt version / message / match type. */}
        <div className="mt-1 flex flex-wrap items-center gap-x-2 font-mono text-[10px] leading-tight text-muted-foreground/70">
          <span
            title={top?.summary_model ?? ""}
            className={previewSource === "summary" ? "text-emerald-600/80" : "text-amber-600/80"}
          >
            {previewSource === "summary"
              ? `summary:${shortModel(top?.summary_model)} · ${ageString(top?.summary_created_at)}`
              : previewSource === "snippet"
                ? "snippet (no summary yet)"
                : "no preview"}
          </span>
          {top?.match_type && <span>match:{top.match_type}</span>}
          {top?.score !== undefined && <span>score:{top.score.toFixed(3)}</span>}
          {top?.message_id && (
            <span className="inline-flex items-center" title={top.message_id}>
              id:{top.message_id.slice(-8)}
              <CopyButton text={top.message_id} label="message id" />
            </span>
          )}
        </div>
      </div>
    </button>
  );
};
