"use client";

import { useEffect, useMemo, useState } from "react";

// Inline viewer for the `get_attachment` tool's batched result. Each
// entry in `results[]` has an `as` field dictating the representation
// we render:
//   - text:            collapsible preformatted block (first 40 lines
//                      until expanded)
//   - inline_pdf:      iframe sourced from a data: URL so the model's
//                      exact bytes render in-browser
//   - inline_image:    <img> from a data: URL
//   - rendered_pages:  list of <img> tags, one per page PNG, with the
//                      page number rendered above each
// The goal is "show the user exactly what the model saw" for an
// attachment fetch. Hidden behind a per-row disclosure so the chat
// transcript stays compact.

type GetAttachmentResult = {
  attachment_id: number;
  as?: string;
  error?: string;
  filename?: string;
  mime_type?: string;
  size_bytes?: number;
  text?: string;
  text_chars?: number;
  quality_note?: string;
  base64?: string;
  total_pages?: number;
  pages?: Array<{ page: number; base64: string; mime_type: string }>;
};

const TEXT_PREVIEW_LINES = 40;
const TEXT_PREVIEW_CHARS = 2000;

const dataUrl = (mime: string, b64: string): string => `data:${mime};base64,${b64}`;

// Decode a base64 string into a Blob so PDFs can render in an iframe
// whose `src` is a `blob:` URL. Two wins over a `data:` URL: no 2MB
// practical soft-limit on URL length, and the iframe runs in an
// isolated origin where `sandbox=""` actually prevents the PDF's
// embedded scripts (XFA, outbound form posts) from running.
const base64ToBlob = (b64: string, mime: string): Blob => {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
};

// React hook: construct a `blob:` URL for the base64 content and
// revoke it on unmount / input change. Keeps the URL stable while the
// caller is mounted so iframes don't 404 mid-session.
const useBlobUrl = (b64: string | undefined, mime: string): string | null => {
  return useMemo(() => {
    if (!b64) return null;
    try {
      return URL.createObjectURL(base64ToBlob(b64, mime));
    } catch {
      return null;
    }
    // Reconstruct when the content changes.
  }, [b64, mime]);
};

const IconFor = ({ mime }: { mime?: string }) => {
  const m = (mime ?? "").toLowerCase();
  let emoji = "📎";
  if (m === "application/pdf") emoji = "📄";
  else if (m.startsWith("image/")) emoji = "🖼";
  else if (m.includes("sheet") || m === "text/csv") emoji = "📊";
  else if (m.includes("word") || m.includes("document")) emoji = "📝";
  return (
    <span aria-hidden className="shrink-0">
      {emoji}
    </span>
  );
};

const Header = ({ r }: { r: GetAttachmentResult }) => {
  const size = r.size_bytes ? `${(r.size_bytes / 1024).toFixed(0)} KB` : null;
  const as = r.as ?? "?";
  const chars = r.text_chars !== undefined ? `${r.text_chars.toLocaleString()} chars` : null;
  const pages = r.pages?.length
    ? `${r.pages.length}${r.total_pages ? `/${r.total_pages}` : ""} pages`
    : null;
  const meta = [as, size, chars, pages].filter(Boolean).join(" · ");
  return (
    <div className="flex items-center gap-1.5 text-[11px] text-neutral-600">
      <IconFor mime={r.mime_type} />
      <span className="font-medium truncate">{r.filename ?? `#${r.attachment_id}`}</span>
      <span className="text-neutral-400">({meta})</span>
    </div>
  );
};

const TextBlock = ({ text }: { text: string }) => {
  const [expanded, setExpanded] = useState(false);
  // Clip by chars first, then by line count. Single-pass produces an
  // accurate "N more chars" footer without double-counting the newline
  // delta between the two steps.
  const clippedByChars = text.slice(0, TEXT_PREVIEW_CHARS);
  const clippedByLines = clippedByChars.split("\n").slice(0, TEXT_PREVIEW_LINES).join("\n");
  const preview = clippedByLines;
  const truncated = preview.length < text.length;
  const body = expanded || !truncated ? text : preview;
  const remaining = text.length - preview.length;
  return (
    <div>
      <pre className="mt-1 max-h-[50vh] overflow-auto whitespace-pre-wrap break-words rounded border border-neutral-200 bg-neutral-50 p-2 text-[11px] leading-relaxed font-mono">
        {body}
        {truncated && !expanded && (
          <span className="text-neutral-400">
            {"\n\n"}... {remaining.toLocaleString()} more chars
          </span>
        )}
      </pre>
      {truncated && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1 text-[11px] text-blue-600 hover:underline"
        >
          {expanded ? "Collapse" : "Show full text"}
        </button>
      )}
    </div>
  );
};

const PdfBlock = ({ r }: { r: GetAttachmentResult }) => {
  const [open, setOpen] = useState(false);
  const src = useBlobUrl(r.base64, r.mime_type ?? "application/pdf");
  // Revoke the blob URL when the component unmounts.
  useEffect(() => {
    return () => {
      if (src) URL.revokeObjectURL(src);
    };
  }, [src]);
  if (!r.base64) return <div className="text-[11px] text-red-600">no bytes</div>;
  if (!src) return <div className="text-[11px] text-red-600">bad base64</div>;
  return (
    <div>
      {open ? (
        <div>
          {/* sandbox="" denies scripts, forms, top-nav, same-origin.
              The browser's built-in PDF viewer still renders (it's not
              an iframe-scripted renderer). Hostile XFA / embedded JS
              in the PDF cannot execute or phone home. */}
          <iframe
            src={src}
            sandbox=""
            className="mt-1 h-[70vh] w-full rounded border border-neutral-200"
            title={r.filename ?? `attachment ${r.attachment_id}`}
          />
          <div className="mt-1 flex gap-3 text-[11px]">
            <a href={src} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
              Open in new tab
            </a>
            <button type="button" onClick={() => setOpen(false)} className="text-neutral-500 hover:underline">
              Collapse
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="mt-1 rounded border border-dashed border-neutral-300 px-3 py-2 text-[11px] text-neutral-600 hover:bg-neutral-50"
        >
          Show PDF inline ({((r.size_bytes ?? 0) / 1024).toFixed(0)} KB)
        </button>
      )}
    </div>
  );
};

const ImageBlock = ({ r }: { r: GetAttachmentResult }) => {
  if (!r.base64) return <div className="text-[11px] text-red-600">no bytes</div>;
  const src = dataUrl(r.mime_type ?? "image/png", r.base64);
  return (
    <img
      src={src}
      alt={r.filename ?? `attachment ${r.attachment_id}`}
      className="mt-1 max-h-[70vh] max-w-full rounded border border-neutral-200"
    />
  );
};

const RenderedPagesBlock = ({ r }: { r: GetAttachmentResult }) => {
  const pages = r.pages ?? [];
  if (pages.length === 0) return <div className="text-[11px] text-red-600">no pages</div>;
  return (
    <div className="mt-1 flex flex-col gap-2">
      {pages.map((p) => (
        <div key={p.page}>
          <div className="text-[10px] uppercase tracking-wide text-neutral-400">page {p.page}</div>
          <img
            src={dataUrl(p.mime_type ?? "image/png", p.base64)}
            alt={`${r.filename ?? "attachment"} page ${p.page}`}
            className="mt-0.5 max-h-[70vh] max-w-full rounded border border-neutral-200"
          />
        </div>
      ))}
    </div>
  );
};

const ResultRow = ({ r }: { r: GetAttachmentResult }) => {
  return (
    <div className="rounded border border-neutral-200 bg-white p-2">
      <Header r={r} />
      {r.quality_note && (
        <div className="mt-1 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
          ⚠ {r.quality_note}
        </div>
      )}
      {r.error ? (
        <div className="mt-1 text-[11px] text-red-600">error: {r.error}</div>
      ) : r.as === "text" ? (
        <TextBlock text={r.text ?? ""} />
      ) : r.as === "inline_pdf" ? (
        <PdfBlock r={r} />
      ) : r.as === "inline_image" ? (
        <ImageBlock r={r} />
      ) : r.as === "rendered_pages" ? (
        <RenderedPagesBlock r={r} />
      ) : (
        <div className="text-[11px] text-neutral-500">unknown representation: {r.as ?? "(missing)"}</div>
      )}
    </div>
  );
};

export const AttachmentInlineViewer = ({ result }: { result: unknown }) => {
  if (!result || typeof result !== "object") return null;
  const o = result as { results?: GetAttachmentResult[]; error?: string };
  if (o.error) return <div className="text-[11px] text-red-600">error: {o.error}</div>;
  const results = o.results ?? [];
  if (results.length === 0) return null;
  return (
    <div className="mt-1 flex flex-col gap-2">
      {results.map((r, i) => (
        <ResultRow key={`${r.attachment_id}-${r.as}-${i}`} r={r} />
      ))}
    </div>
  );
};
