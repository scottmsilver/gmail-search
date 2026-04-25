"use client";

import { useEffect, useState } from "react";

import { MarkdownText } from "./MarkdownText";
import { PdfRenderer } from "./PdfRenderer";

// A mime-type dispatcher. Given a URL + mime, render the right view:
//   application/pdf   → <PdfRenderer>
//   image/*           → <img>
//   text/csv          → a simple HTML table (first ~200 rows)
//   text/markdown     → MarkdownText
//   text/* or JSON    → preformatted text
//   anything else     → Download link
//
// We intentionally keep this lean — cc-web's file-viewer is 840 lines
// because it handles XLSX, EML, thread JSON, etc. We ship the core
// cases here and add more as the need shows up.

type Props = {
  url: string;
  filename: string;
  mimeType: string;
};

const isTextLike = (m: string) =>
  m.startsWith("text/") || m === "application/json" || m === "application/javascript";

export const FileViewer = ({ url, filename, mimeType }: Props) => {
  if (mimeType === "application/pdf") {
    return <PdfRenderer url={url} />;
  }
  if (mimeType.startsWith("image/")) {
    return (
      <div className="flex h-full items-center justify-center overflow-auto bg-neutral-100 p-4">
        <img
          src={url}
          alt={filename}
          className="max-h-full max-w-full rounded border border-neutral-200 bg-white shadow-md"
        />
      </div>
    );
  }
  if (mimeType === "text/csv") return <CsvTable url={url} />;
  if (mimeType === "text/markdown") return <MarkdownFromUrl url={url} />;
  if (isTextLike(mimeType)) return <TextBody url={url} />;

  return (
    <div className="p-6 text-sm">
      <p className="text-neutral-700">
        No inline preview for <span className="font-medium">{mimeType}</span>.
      </p>
      <a
        href={url}
        download={filename}
        className="mt-3 inline-block text-blue-600 underline"
      >
        Download {filename}
      </a>
    </div>
  );
};

// ── Text-like renderers ────────────────────────────────────────────

const useText = (url: string): { body: string | null; err: string | null } => {
  const [body, setBody] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setBody(null);
    setErr(null);
    fetch(url)
      .then(async (r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.text();
      })
      .then((t) => {
        if (!cancelled) setBody(t);
      })
      .catch((e) => {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [url]);
  return { body, err };
};

const TextBody = ({ url }: { url: string }) => {
  const { body, err } = useText(url);
  if (err) return <div className="p-6 text-sm text-red-600">Error: {err}</div>;
  if (body === null) return <div className="p-6 text-sm text-neutral-500">Loading…</div>;
  return (
    <pre className="h-full overflow-auto p-4 text-xs font-mono leading-5 whitespace-pre-wrap break-words">
      {body}
    </pre>
  );
};

const MarkdownFromUrl = ({ url }: { url: string }) => {
  const { body, err } = useText(url);
  if (err) return <div className="p-6 text-sm text-red-600">Error: {err}</div>;
  if (body === null) return <div className="p-6 text-sm text-neutral-500">Loading…</div>;
  return (
    <div className="overflow-auto p-6 prose prose-sm max-w-none">
      <MarkdownText text={body} />
    </div>
  );
};

// ── CSV table (first N rows) ───────────────────────────────────────

// Naive CSV split. Handles the common "field,field,field" case and
// double-quoted fields with embedded commas. For anything fancier
// (newlines inside quotes, escaped quotes) we let the row fall
// through as-is — this is a preview, not an ETL tool.
const splitCsvLine = (line: string): string[] => {
  const out: string[] = [];
  let cur = "";
  let inQuote = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuote = !inQuote;
      continue;
    }
    if (ch === "," && !inQuote) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
};

const CSV_ROW_CAP = 200;

const CsvTable = ({ url }: { url: string }) => {
  const { body, err } = useText(url);
  if (err) return <div className="p-6 text-sm text-red-600">Error: {err}</div>;
  if (body === null) return <div className="p-6 text-sm text-neutral-500">Loading CSV…</div>;
  const lines = body.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length === 0) return <div className="p-6 text-sm text-neutral-500">Empty CSV.</div>;
  const header = splitCsvLine(lines[0]);
  const rows = lines.slice(1, 1 + CSV_ROW_CAP).map(splitCsvLine);
  const truncated = lines.length - 1 > CSV_ROW_CAP;
  return (
    <div className="h-full overflow-auto">
      <table className="min-w-full text-xs border-collapse">
        <thead className="bg-neutral-100 sticky top-0">
          <tr>
            {header.map((h, i) => (
              <th
                key={i}
                className="text-left font-semibold px-3 py-1.5 border-b border-neutral-200 whitespace-nowrap"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri} className="border-b border-neutral-100">
              {r.map((c, ci) => (
                <td key={ci} className="px-3 py-1 align-top font-mono whitespace-pre-wrap">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {truncated && (
        <div className="px-3 py-2 text-[11px] text-neutral-500 bg-neutral-50 border-t">
          Showing first {CSV_ROW_CAP} rows of {lines.length - 1}. Download for full CSV.
        </div>
      )}
    </div>
  );
};
