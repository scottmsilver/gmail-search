"use client";

import { useEffect, useRef, useState } from "react";

import { usePreview } from "./PreviewContext";

// Citation chip for `[art:<id>]` — an Analyst-produced plot, CSV,
// or HTML blob served by /api/artifact/<id>. On first render we GET
// and peek the headers to learn content-type + filename, then on
// click we open the shared PreviewDrawer (PDFs, images, CSV, text,
// markdown). No more inline toggle — clicking always opens the same
// side panel the rest of the app uses.

type Props = {
  artifactId: number;
};

type ArtifactMeta = {
  mimeType: string;
  filename: string;
};

const fetchMeta = async (id: number): Promise<ArtifactMeta> => {
  // HEAD isn't wired upstream yet; do a tiny GET and peek only the
  // headers. Body is discarded via resp.body.cancel() below.
  const resp = await fetch(`/api/artifact/${encodeURIComponent(String(id))}`, {
    method: "GET",
  });
  if (!resp.ok) throw new Error(`artifact ${id}: ${resp.status}`);
  const ct = resp.headers.get("content-type") ?? "application/octet-stream";
  const cd = resp.headers.get("content-disposition") ?? "";
  const match = /filename="?([^";]+)"?/.exec(cd);
  const filename = match?.[1] ?? `artifact-${id}`;
  await resp.body?.cancel();
  return { mimeType: ct, filename };
};

export const ArtifactChip = ({ artifactId }: Props) => {
  const [meta, setMeta] = useState<ArtifactMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetchedRef = useRef(false);
  const { openPreview } = usePreview();

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;
    fetchMeta(artifactId)
      .then(setMeta)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [artifactId]);

  const href = `/api/artifact/${encodeURIComponent(String(artifactId))}`;
  const label = meta?.filename ?? `art:${artifactId}`;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!meta) return;
    openPreview({
      url: href,
      filename: meta.filename,
      mimeType: meta.mimeType,
      kind: "artifact",
    });
  };

  return (
    <span className="inline-flex items-center gap-1 align-baseline">
      <button
        type="button"
        onClick={handleClick}
        disabled={!meta}
        className="inline-flex max-w-[28ch] items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-xs text-emerald-900 transition-colors hover:bg-emerald-100 disabled:cursor-wait disabled:opacity-60"
        title={meta ? `${meta.filename} · ${meta.mimeType}` : `artifact ${artifactId}`}
      >
        <span aria-hidden>📈</span>
        <span className="truncate font-medium">{label}</span>
      </button>
      {error && <span className="text-xs text-red-600">({error})</span>}
    </span>
  );
};
