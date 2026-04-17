"use client";

import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { useMessage } from "@assistant-ui/react";

import { extractThreadHints } from "@/lib/extractThreadHints";
import { linkifyRefs, REF_PREFIX } from "@/lib/linkifyRefs";

import { CitationChip } from "./CitationChip";
import { useThreadDrawer } from "./ThreadDrawerContext";

export const MarkdownText = ({ text }: { text: string }) => {
  const { setOpenThreadId } = useThreadDrawer();
  const parts = useMessage((m) => m.content) as readonly { type: string; result?: unknown }[];
  const hints = extractThreadHints(parts);
  const knownIds = hints.map((h) => h.thread_id);

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
      return (
        <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline" {...rest}>
          {children}
        </a>
      );
    },
  };

  return (
    <div className="prose prose-sm max-w-4xl prose-p:my-1.5 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0 prose-headings:my-2 prose-pre:my-2 prose-code:before:content-none prose-code:after:content-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={components}
        urlTransform={(url) => url}
      >
        {linkifyRefs(text, knownIds)}
      </ReactMarkdown>
    </div>
  );
};
