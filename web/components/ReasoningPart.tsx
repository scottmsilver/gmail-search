"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const Caret = ({ open }: { open: boolean }) => (
  <svg
    className={`w-3 h-3 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
  </svg>
);

type Props = {
  text: string;
  status: { type: "running" | "complete" | "incomplete" | string };
};

const stripMarkdownForPreview = (raw: string): string =>
  raw
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/[#*_`>~|]/g, "")
    .replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();

export const ReasoningPart = ({ text, status }: Props) => {
  const isThinking = status.type === "running";
  const [open, setOpen] = useState(false);

  if (!text && !isThinking) return null;

  return (
    <div className="my-1 text-xs text-neutral-500">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 hover:text-neutral-800"
      >
        <Caret open={open} />
        <span className="italic">{isThinking ? "Thinking…" : "Thoughts"}</span>
        {!open && text && (
          <span className="text-neutral-400 truncate max-w-[60ch]">— {stripMarkdownForPreview(text).slice(0, 80)}…</span>
        )}
      </button>
      {open && text && (
        <div className="mt-1 ml-4 border-l-2 border-neutral-200 pl-3 text-[12px] leading-relaxed text-neutral-600 prose prose-sm max-w-none prose-p:my-1 prose-headings:my-1.5 prose-headings:text-neutral-700 prose-strong:text-neutral-700 prose-li:my-0 prose-ul:my-1 prose-ol:my-1">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        </div>
      )}
    </div>
  );
};
