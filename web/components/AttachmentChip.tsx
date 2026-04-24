"use client";

import { useEffect, useState, useSyncExternalStore } from "react";

import {
  fetchAttachmentMeta,
  getCachedAttachment,
  subscribeAttachment,
} from "@/lib/attachmentCache";

export type AttachmentHint = {
  attachment_id: number;
  filename?: string;
  mime_type?: string;
  thread_id?: string;
};

type Props = {
  attachmentId: number;
  hints: AttachmentHint[];
  onOpenThread: (threadId: string) => void;
};

const ICONS: Record<string, string> = {
  pdf: "📄",
  image: "🖼",
  spreadsheet: "📊",
  document: "📝",
  archive: "🗜",
  default: "📎",
};

const pickIcon = (mime: string | undefined): string => {
  if (!mime) return ICONS.default;
  if (mime === "application/pdf") return ICONS.pdf;
  if (mime.startsWith("image/")) return ICONS.image;
  if (mime.includes("spreadsheet") || mime.includes("excel") || mime === "text/csv") return ICONS.spreadsheet;
  if (mime.includes("word") || mime.includes("document")) return ICONS.document;
  if (mime.includes("zip") || mime.includes("archive")) return ICONS.archive;
  return ICONS.default;
};

const findInHints = (id: number, hints: AttachmentHint[]) => hints.find((h) => h.attachment_id === id) ?? null;

const useCachedAttachment = (id: number) =>
  useSyncExternalStore(
    (cb) => subscribeAttachment(id, cb),
    () => getCachedAttachment(id),
    () => undefined,
  );

export const AttachmentChip = ({ attachmentId, hints, onOpenThread }: Props) => {
  const cached = useCachedAttachment(attachmentId);
  const [didFetch, setDidFetch] = useState(false);

  useEffect(() => {
    if (findInHints(attachmentId, hints)) return;
    if (didFetch) return;
    if (cached && cached !== "error") return;
    setDidFetch(true);
    void fetchAttachmentMeta(attachmentId);
  }, [attachmentId, hints, cached, didFetch]);

  const fromHints = findInHints(attachmentId, hints);
  const fromCache = cached && cached !== "loading" && cached !== "error" ? cached : null;

  // Resolve displayable fields, falling back in order: tool hints (best,
  // zero extra round trip) → the lazy /api/attachment/<id>/meta fetch →
  // a placeholder while loading.
  const filename = fromHints?.filename ?? fromCache?.filename ?? null;
  const mime = fromHints?.mime_type ?? fromCache?.mime_type ?? null;
  const threadId = fromHints?.thread_id ?? fromCache?.thread_id ?? null;
  const isLoading = !filename && cached === "loading";
  const failed = cached === "error" && !filename;

  const label = filename ?? (isLoading ? "Loading…" : failed ? `#${attachmentId} (not found)` : `#${attachmentId}`);

  const handleClick = () => {
    if (threadId) onOpenThread(threadId);
  };

  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        handleClick();
      }}
      disabled={!threadId}
      className="inline-flex max-w-[28ch] items-center gap-1.5 rounded-md border border-amber-100 bg-amber-50 px-2 py-0.5 align-baseline text-xs text-amber-900 transition-colors hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-60"
      title={filename ? `${filename}${mime ? ` · ${mime}` : ""}` : `Attachment ${attachmentId}`}
    >
      <span aria-hidden className="shrink-0">
        {pickIcon(mime ?? undefined)}
      </span>
      <span className="truncate font-medium">{label}</span>
    </button>
  );
};
