"use client";

import { useEffect, useRef, useState } from "react";

// Citation chip for `[art:<id>]` — an Analyst-produced plot, CSV,
// or HTML blob served by /api/artifact/<id>. On first render we HEAD
// the endpoint to learn the content type, then on click toggle a
// preview:
//   image/*  → inline <img>
//   text/csv → first-10-lines table preview + Download button
//   other    → Download button
//
// Server returns Content-Type + Content-Disposition headers verbatim,
// so we don't duplicate mime detection here — just trust what the
// artifact row carries.

type Props = {
  artifactId: number;
};

type ArtifactMeta = {
  mimeType: string;
  filename: string;
};

const fetchMeta = async (id: number): Promise<ArtifactMeta> => {
  // HEAD isn't wired upstream yet; do a tiny GET and peek only the
  // headers. Body is fine to discard — we revoke its consumption.
  const resp = await fetch(`/api/artifact/${encodeURIComponent(String(id))}`, {
    method: "GET",
  });
  if (!resp.ok) throw new Error(`artifact ${id}: ${resp.status}`);
  const ct = resp.headers.get("content-type") ?? "application/octet-stream";
  const cd = resp.headers.get("content-disposition") ?? "";
  const match = /filename="?([^";]+)"?/.exec(cd);
  const filename = match?.[1] ?? `artifact-${id}`;
  // Release body we don't need right now to free the connection.
  await resp.body?.cancel();
  return { mimeType: ct, filename };
};

export const ArtifactChip = ({ artifactId }: Props) => {
  const [meta, setMeta] = useState<ArtifactMeta | null>(null);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;
    fetchMeta(artifactId)
      .then(setMeta)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [artifactId]);

  const href = `/api/artifact/${encodeURIComponent(String(artifactId))}`;
  const label = meta?.filename ?? `art:${artifactId}`;
  const isImage = meta?.mimeType.startsWith("image/");
  const isCsv = meta?.mimeType === "text/csv";

  return (
    <span className="inline-flex items-center gap-1 align-baseline">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="inline-flex max-w-[28ch] items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-xs text-emerald-900 transition-colors hover:bg-emerald-100"
        title={meta ? `${meta.filename} · ${meta.mimeType}` : `artifact ${artifactId}`}
      >
        <span aria-hidden>📈</span>
        <span className="truncate font-medium">{label}</span>
      </button>
      {open && meta && !error && (
        <span className="ml-1 inline-block align-top">
          {isImage && (
            <img
              src={href}
              alt={meta.filename}
              className="mt-1 max-h-[60vh] max-w-full rounded border border-neutral-200"
            />
          )}
          {!isImage && (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-600 underline"
              onClick={(e) => e.stopPropagation()}
            >
              Open {isCsv ? "CSV" : "file"}
            </a>
          )}
        </span>
      )}
      {error && <span className="text-xs text-red-600">({error})</span>}
    </span>
  );
};
