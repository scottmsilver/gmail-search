"use client";

import { useState } from "react";

import { CitableMarkdown } from "@/components/CitableMarkdown";
import { MiddleTruncate } from "@/components/MiddleTruncate";
import type { SearchThread } from "@/lib/backend";
import { formatSmartDate } from "@/lib/datetime";
import { useShowSearchDebug } from "@/lib/prefs";
import { cleanSender } from "@/lib/sender";

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

// Summaries frequently restate the sender's name as their first
// subject ("Scott Silver says X", "Jesica Peterson asks Y") — which
// is redundant because the sender column right above the summary
// already shows the same name. If the summary starts with a prefix
// that matches the cleaned sender name (case-insensitive), drop it
// and re-capitalize the next word so the sentence still reads as a
// sentence: "Scott Silver says X" → "Says X".
//
// We try several candidate prefixes derived from the sender header
// so that noise like credentials or vendor-via suffixes doesn't
// block the match:
//   "Sasha Torres, MA, BCBA"           → also try "Sasha Torres"
//   "San Bruno Flower Fashions (via X)" → also try "San Bruno Flower Fashions"
//
// Possessives ("Scott Silver's reply…") and partial-name phrases
// ("Scott and Joy met…") are left alone because they aren't
// redundant — the name is load-bearing there.
const _senderPrefixCandidates = (name: string): string[] => {
  const out = new Set<string>();
  const add = (s: string) => {
    const trimmed = s.trim();
    if (trimmed.length >= 2) out.add(trimmed);
  };
  add(name);
  // Strip credentials after a comma: "Sasha Torres, MA, BCBA" → "Sasha Torres"
  const commaIdx = name.indexOf(",");
  if (commaIdx > 0) add(name.slice(0, commaIdx));
  // Strip "(via X)" / parenthetical suffix.
  const parenIdx = name.indexOf("(");
  if (parenIdx > 0) add(name.slice(0, parenIdx));
  // Also the combination of both trims.
  if (parenIdx > 0 && commaIdx > 0) {
    add(name.slice(0, Math.min(parenIdx, commaIdx)));
  }
  // Sort longest-first so we always strip the MOST specific match.
  return [...out].sort((a, b) => b.length - a.length);
};

const stripRedundantSenderPrefix = (summary: string, fromAddr: string): string => {
  const base = cleanSender(fromAddr).trim();
  if (!base) return summary;
  for (const cand of _senderPrefixCandidates(base)) {
    if (summary.length <= cand.length) continue;
    if (summary.slice(0, cand.length).toLowerCase() !== cand.toLowerCase()) continue;
    const nextChar = summary.charAt(cand.length);
    // Must be a word-break — space / tab — not a letter or a digit.
    if (nextChar !== " " && nextChar !== "\t") continue;
    const rest = summary.slice(cand.length).trimStart();
    // Possessive — keep the name.
    if (rest.startsWith("'s ") || rest.startsWith("’s ")) return summary;
    if (!rest) continue;
    return rest.charAt(0).toUpperCase() + rest.slice(1);
  }
  return summary;
};

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

// Gmail-style chevron indicator shown next to the sender column when
// the user was involved in the thread. A single `›` (sent-to or
// replied) is what we render today. Gmail distinguishes `›` (user
// in To/Cc alongside others) from `»` (user was the SOLE To
// recipient); getting that right needs a server-side comparison of
// the user's email against each matched message's to_addr list.
// Flagged for follow-up.
const SenderChevron = () => (
  <span
    aria-label="You are on this thread"
    title="You are on this thread"
    className="mr-1 select-none font-semibold text-muted-foreground"
  >
    ›
  </span>
);

// Little external-link opener — rendered as <span role="button"> for
// the same reason CopyButton is: the outer row is already a <button>,
// and HTML forbids nested buttons. Opens the URL in a new tab on
// plain click; a shift-click opens a sized popup window instead for
// users who prefer a floating Gmail window. Iframe-inline isn't
// possible — Gmail sends `X-Frame-Options: DENY`, so the browser
// refuses to render it.
const OpenLinkButton = ({
  url,
  label,
  popupName,
}: {
  url: string;
  label: string;
  popupName?: string;
}) => {
  const onActivate = (e: React.SyntheticEvent) => {
    e.stopPropagation();
    e.preventDefault();
    const native = (e as unknown as React.MouseEvent).shiftKey;
    if (native && popupName) {
      window.open(url, popupName, "popup,width=1100,height=800,noopener,noreferrer");
    } else {
      window.open(url, "_blank", "noopener,noreferrer");
    }
  };
  return (
    <span
      role="button"
      tabIndex={0}
      aria-label={`Open ${label}`}
      title={`Open ${label}${popupName ? " (shift-click for popup)" : ""}`}
      onClick={onActivate}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onActivate(e);
      }}
      className="ml-1 inline-flex h-4 w-4 cursor-pointer items-center justify-center rounded text-muted-foreground/60 hover:bg-muted hover:text-foreground"
    >
      {/* External-link glyph: box with an arrow pointing out the top-right. */}
      <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth={2}>
        <path d="M14 5h5v5" />
        <path d="M19 5L10 14" />
        <path d="M19 13v5a1 1 0 01-1 1H6a1 1 0 01-1-1V6a1 1 0 011-1h5" />
      </svg>
    </span>
  );
};

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
  const [showDebug] = useShowSearchDebug();
  const hasSummary = !!(top?.summary && top.summary.trim());
  // Prefer the LLM-generated summary over the raw matched snippet;
  // fall back to the cleaned snippet when no summary exists. When
  // the summary starts with the sender's name (redundant with the
  // sender column rendered right above it), strip the name prefix.
  const rawPreview = hasSummary ? top!.summary!.trim() : cleanSnippet(top?.snippet ?? "").slice(0, 180);
  const preview =
    hasSummary && top?.from_addr ? stripRedundantSenderPrefix(rawPreview, top.from_addr) : rawPreview;
  const previewSource: "summary" | "snippet" | "empty" = hasSummary
    ? "summary"
    : preview
      ? "snippet"
      : "empty";
  const hasAttachment = thread.matches.some((m) => m.attachment_filename);

  // Single-row grid:
  //   sender-names · subject · date
  // plus a row 2 below that holds the summary + (optional) debug
  // footer, spanning all three columns. The avatar-initials circle
  // was removed — it added colour but no information.
  // We can't use <button> as the outer because summaries now render
  // markdown, which embeds <a> elements — nesting an <a> inside a
  // <button> is invalid HTML (and some browsers swallow the click).
  // So we use role=button + keyboard handling instead.
  const handleActivate = () => onOpen(thread.thread_id);
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleActivate}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleActivate();
        }
      }}
      className="group grid w-full cursor-pointer grid-cols-[200px_1fr_auto] items-start gap-x-3 gap-y-1 border-b px-4 py-2.5 text-left text-sm transition-colors hover:bg-accent/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {/* Names (row 1, col 1) */}
      <div className="min-w-0 truncate pt-0.5">
        {thread.user_replied && <SenderChevron />}
        <span className="font-medium text-foreground">{shortPeople(thread.participants)}</span>
        {thread.message_count > 1 && (
          <span className="ml-1.5 text-xs font-normal text-muted-foreground">{thread.message_count}</span>
        )}
      </div>

      {/* Subject (row 1, col 2) — middle-ellipsised so Gmail's
          "Re: Fwd: Fwd: Re: real subject [EXT] [EXT] [EXT]" threads
          keep the meaningful middle visible instead of getting cut
          off on the right like end-truncate did. */}
      <div className="flex min-w-0 items-center gap-1 font-medium text-foreground">
        {hasAttachment && <PaperclipIcon />}
        <MiddleTruncate text={thread.subject} className="min-w-0 flex-1" />
        {/* Open in Gmail — new tab. Shift-click for popup window. */}
        <OpenLinkButton
          url={`https://mail.google.com/mail/u/0/#all/${thread.thread_id}`}
          label="in Gmail"
          popupName="gmail-popup"
        />
      </div>

      {/* Date (row 1, col 4) */}
      <div className="shrink-0 pt-1 text-xs text-muted-foreground">{formatSmartDate(thread.date_last)}</div>

      {/* Summary + debug footer (row 2, spans names+subject+date) */}
      {/* Summary row (row 2, spanning all three columns now that
          the avatar column is gone). */}
      <div className="col-start-1 col-span-3 min-w-0">
        {preview && (
          <div className="whitespace-normal break-words text-muted-foreground">
            {previewSource === "summary" ? (
              <CitableMarkdown text={preview} hints={[]} variant="inline" />
            ) : (
              preview
            )}
          </div>
        )}
        {/* Debug footer — only rendered when the user flips on the
            "Show search debug" toggle in Settings. Lives in
            localStorage via `useShowSearchDebug`. */}
        {showDebug && (
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
        )}
      </div>
    </div>
  );
};
