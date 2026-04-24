"use client";

import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { ART_PREFIX, ATT_PREFIX, linkifyRefs, REF_PREFIX } from "@/lib/linkifyRefs";

import { ArtifactChip } from "./ArtifactChip";
import { AttachmentChip, type AttachmentHint } from "./AttachmentChip";
import { CitationChip, type ThreadHint } from "./CitationChip";
import { useThreadDrawer } from "./ThreadDrawerContext";

type Props = {
  text: string;
  hints: ThreadHint[];
  // Optional attachment hints — tool results that list attachments
  // pre-populate this so `[att:<id>]` citations render a fully-labeled
  // chip without an extra fetch. Missing hint → chip falls back to the
  // lazy /api/attachment/<id>/meta lookup.
  attHints?: AttachmentHint[];
  // "prose" wraps the output in a Tailwind prose container capped at
  // max-w-4xl — right for chat messages. "inline" drops the wrapper
  // so the parent controls width (used by search result rows where
  // the summary needs to span the full row, no max-width).
  variant?: "prose" | "inline";
};

// Lower-level shared markdown + citation renderer. Both MarkdownText
// (context-based hints) and BattleSide (prop-based hints from its own
// tool results) delegate to this so the rendering logic lives in one
// place.
export const CitableMarkdown = ({ text, hints, attHints, variant = "prose" }: Props) => {
  const { setOpenThreadId } = useThreadDrawer();
  const knownIds = hints.map((h) => h.thread_id);
  const attList = attHints ?? [];

  const components: Components = {
    a: ({ href, children, ...rest }) => {
      if (href?.startsWith(REF_PREFIX)) {
        return (
          <CitationChip
            threadId={href.slice(REF_PREFIX.length)}
            hints={hints}
            onOpen={setOpenThreadId}
          />
        );
      }
      if (href?.startsWith(ATT_PREFIX)) {
        const idStr = href.slice(ATT_PREFIX.length);
        const id = parseInt(idStr, 10);
        if (Number.isFinite(id)) {
          return (
            <AttachmentChip attachmentId={id} hints={attList} onOpenThread={setOpenThreadId} />
          );
        }
      }
      if (href?.startsWith(ART_PREFIX)) {
        const idStr = href.slice(ART_PREFIX.length);
        const id = parseInt(idStr, 10);
        if (Number.isFinite(id)) {
          return <ArtifactChip artifactId={id} />;
        }
      }
      // In "inline" (summary) mode, suppress mailto autolinks — the
      // LLM frequently mentions the sender's email in its output, and
      // remark-gfm happily turns "alice@example.com" into a mailto:
      // link. That's noise in a summary row (the sender chip above
      // the summary already shows the address). Render as plain text.
      if (variant === "inline" && href?.startsWith("mailto:")) {
        return <>{children}</>;
      }
      // When the visible text IS a URL (either auto-linked from bare
      // prose or wrapped in `[long-url](long-url)`), swap it for a
      // short host-based label so a 900-char tracking URL doesn't take
      // over the row. The href keeps the full URL intact for clicking.
      const displayChildren = shortenIfUrlLabel(children, href);
      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 underline"
          onClick={(e) => e.stopPropagation()}
          title={href}
          {...rest}
        >
          {displayChildren}
        </a>
      );
    },
  };

  const wrapperClass =
    variant === "prose"
      ? "prose prose-sm max-w-4xl prose-p:my-1.5 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0 prose-headings:my-2 prose-pre:my-2 prose-code:before:content-none prose-code:after:content-none"
      // inline: no prose, no max-width — parent row controls layout.
      // "[&_p]:m-0" so ReactMarkdown's default <p> wrapper doesn't
      // inject vertical margin into a single-line summary.
      : "[&_p]:m-0 [&_p]:inline";
  return (
    <div className={wrapperClass}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components} urlTransform={safeUrl}>
        {linkifyRefs(text, knownIds)}
      </ReactMarkdown>
    </div>
  );
};

// We pass through ref:// (citation chips) and otherwise allow http(s)/mailto
// only. ReactMarkdown's default urlTransform strips most XSS schemes, but we
// were overriding it with `(url) => url` to keep ref:// links from being
// mangled — that opened javascript:/data:/vbscript: as side effects.
const SAFE_SCHEMES = /^(?:https?|mailto|ref|att|art):/i;
const safeUrl = (url: string): string => {
  if (!url) return "";
  // Relative URLs (no scheme) are fine.
  if (!/^[a-z][a-z0-9+.-]*:/i.test(url)) return url;
  return SAFE_SCHEMES.test(url) ? url : "";
};

// Turn a raw URL into a readable label: "host.com/firstSegment" capped
// at 60 chars. If parsing fails, fall back to a truncated head.
const URL_LABEL_CAP = 60;
export const prettyUrlLabel = (raw: string): string => {
  try {
    const u = new URL(raw);
    const host = u.hostname.replace(/^www\./, "");
    const firstSeg = u.pathname.split("/").filter(Boolean)[0] ?? "";
    const label = firstSeg ? `${host}/${firstSeg}` : host;
    return label.length > URL_LABEL_CAP ? `${label.slice(0, URL_LABEL_CAP - 1)}…` : label;
  } catch {
    return raw.length > URL_LABEL_CAP ? `${raw.slice(0, URL_LABEL_CAP - 1)}…` : raw;
  }
};

// When a markdown anchor's visible text is a URL, it's almost always
// the same URL that's in the href (LLM wrote a bare URL OR wrapped the
// URL in itself: `[url](url)`). Short URLs read fine as-is — but
// tracking URLs with JWT-sized query strings dominate the row. This
// helper returns a shortened label in those cases, otherwise passes
// children through untouched so legitimate short labels aren't
// rewritten.
const URL_VISIBLE_MAX = 60;
const extractTextFromChildren = (children: ReactNode): string | null => {
  if (typeof children === "string") return children;
  if (Array.isArray(children) && children.every((c) => typeof c === "string")) {
    return children.join("");
  }
  return null;
};

const shortenIfUrlLabel = (children: ReactNode, href: string | undefined): ReactNode => {
  if (!href) return children;
  const text = extractTextFromChildren(children);
  if (text === null) return children;
  const looksLikeUrl = /^https?:\/\//i.test(text);
  if (!looksLikeUrl) return children;
  if (text.length <= URL_VISIBLE_MAX) return children;
  return prettyUrlLabel(href);
};
