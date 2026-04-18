"use client";

import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { linkifyRefs, REF_PREFIX } from "@/lib/linkifyRefs";

import { CitationChip, type ThreadHint } from "./CitationChip";
import { useThreadDrawer } from "./ThreadDrawerContext";

type Props = {
  text: string;
  hints: ThreadHint[];
};

// Lower-level shared markdown + citation renderer. Both MarkdownText
// (context-based hints) and BattleSide (prop-based hints from its own
// tool results) delegate to this so the rendering logic lives in one
// place.
export const CitableMarkdown = ({ text, hints }: Props) => {
  const { setOpenThreadId } = useThreadDrawer();
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
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components} urlTransform={(url) => url}>
        {linkifyRefs(text, knownIds)}
      </ReactMarkdown>
    </div>
  );
};
