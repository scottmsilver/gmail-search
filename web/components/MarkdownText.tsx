"use client";

import { useMessage } from "@assistant-ui/react";

import { extractCitationHints } from "@/lib/extractThreadHints";

import { CitableMarkdown } from "./CitableMarkdown";

// Thin wrapper over CitableMarkdown that pulls citation hints from the
// current assistant-ui message context. Battle-mode messages use
// CitableMarkdown directly with their own hints since their tool
// results live in a data-battle part, not message-level tool-calls.
export const MarkdownText = ({ text }: { text: string }) => {
  const parts = useMessage((m) => m.content) as readonly { type: string; result?: unknown }[];
  const { threads, attachments } = extractCitationHints(parts);
  return <CitableMarkdown text={text} hints={threads} attHints={attachments} />;
};
