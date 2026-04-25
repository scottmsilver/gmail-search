"use client";

import { useEffect, useRef, useState } from "react";

// Minimal pdfjs doc/page types — matches what the preview uses without
// pulling the full pdfjs-dist type graph into every consumer.
type PdfPageViewport = { width: number; height: number };
type PdfPage = {
  getViewport: (o: { scale: number }) => PdfPageViewport;
  render: (o: { canvasContext: CanvasRenderingContext2D; viewport: PdfPageViewport }) => {
    promise: Promise<void>;
  };
};
type PdfDocument = {
  numPages: number;
  getPage: (n: number) => Promise<PdfPage>;
};

// Render one page of a PDF onto a canvas. `url` is same-origin; pdfjs
// fetches the bytes itself. We lazy-import pdfjs-dist so the chunk
// only loads when the drawer actually opens a PDF — no cost on first
// paint of the app.
export const PdfRenderer = ({ url }: { url: string }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [doc, setDoc] = useState<PdfDocument | null>(null);
  const [page, setPage] = useState(1);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Load the document when `url` changes. pdfjs-dist needs its worker
  // served as a separate file — see public/pdf.worker.min.mjs (copied
  // from node_modules/pdfjs-dist/build/... in setup).
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);
    setPage(1);
    (async () => {
      try {
        const pdfjs = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
        const loaded = (await pdfjs.getDocument(url).promise) as unknown as PdfDocument;
        if (cancelled) return;
        setDoc(loaded);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [url]);

  // Rerender the current page whenever doc or page index changes. We
  // target 1.5× scale — readable on typical displays without bloating
  // memory on long docs.
  useEffect(() => {
    if (!doc || !canvasRef.current) return;
    let cancelled = false;
    (async () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const p = await doc.getPage(page);
      if (cancelled) return;
      const viewport = p.getViewport({ scale: 1.5 });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      await p.render({ canvasContext: ctx, viewport }).promise;
    })();
    return () => {
      cancelled = true;
    };
  }, [doc, page]);

  if (loading) return <div className="p-6 text-sm text-neutral-500">Loading PDF…</div>;
  if (err) return <div className="p-6 text-sm text-red-600">PDF error: {err}</div>;
  if (!doc) return null;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b px-3 py-1.5 text-xs bg-neutral-50">
        <button
          type="button"
          className="rounded px-1.5 py-0.5 hover:bg-neutral-200 disabled:opacity-40"
          disabled={page <= 1}
          onClick={() => setPage((p) => Math.max(1, p - 1))}
        >
          ←
        </button>
        <span>
          Page {page} / {doc.numPages}
        </span>
        <button
          type="button"
          className="rounded px-1.5 py-0.5 hover:bg-neutral-200 disabled:opacity-40"
          disabled={page >= doc.numPages}
          onClick={() => setPage((p) => Math.min(doc.numPages, p + 1))}
        >
          →
        </button>
      </div>
      <div className="flex-1 overflow-auto bg-neutral-100 p-4">
        <canvas ref={canvasRef} className="mx-auto block shadow-md" />
      </div>
    </div>
  );
};
