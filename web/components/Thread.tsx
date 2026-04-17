"use client";

import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";

import { DebugIdBadge } from "./DebugIdBadge";
import { MarkdownText } from "./MarkdownText";
import { ModelPicker } from "./ModelPicker";
import { ReasoningPart } from "./ReasoningPart";
import { ToolCallUI } from "./ToolCallUI";

const SEND_ICON = (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);

const TextPart = ({ text }: { text: string }) => <MarkdownText text={text} />;

const UserMessage = () => (
  <MessagePrimitive.Root className="flex justify-end my-4 px-4">
    <div className="max-w-[80%] bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-2 whitespace-pre-wrap">
      <MessagePrimitive.Parts components={{ Text: ({ text }) => <>{text}</> }} />
    </div>
  </MessagePrimitive.Root>
);

type DebugIdData = { id: string; log_path: string };
type CitationWarningData = { broken: string[]; message: string };

const DebugIdPart = ({ data }: { data: DebugIdData }) => <DebugIdBadge data={data} />;
const CitationWarningPart = ({ data }: { data: CitationWarningData }) => (
  <div className="my-2 px-3 py-2 rounded border border-amber-300 bg-amber-50 text-xs text-amber-800">
    ⚠ {data.message} ({data.broken.join(", ")})
  </div>
);

const AssistantMessage = () => (
  <MessagePrimitive.Root className="my-4 px-4">
    <div className="max-w-[95%]">
      <MessagePrimitive.Parts
        components={{
          Text: TextPart,
          Reasoning: ReasoningPart,
          tools: { Fallback: ToolCallUI },
          data: {
            by_name: {
              "debug-id": DebugIdPart as never,
              "citation-warning": CitationWarningPart as never,
            },
          },
        }}
      />
    </div>
  </MessagePrimitive.Root>
);

const Composer = () => (
  <ComposerPrimitive.Root className="px-4 pb-4 pt-1 bg-white">
    <ModelPicker />
    <div className="mt-2 flex gap-2 rounded-2xl border border-neutral-200 bg-neutral-50 focus-within:border-neutral-400 focus-within:bg-white transition-colors px-3 py-2">
      <ComposerPrimitive.Input
        placeholder="Ask about your email…"
        className="flex-1 bg-transparent focus:outline-none resize-none placeholder:text-neutral-400 text-sm leading-6"
        rows={1}
        autoFocus
      />
      <ComposerPrimitive.Send
        aria-label="Send"
        className="self-end rounded-full bg-neutral-900 text-white w-8 h-8 disabled:opacity-30 flex items-center justify-center hover:bg-neutral-700 transition-colors"
      >
        {SEND_ICON}
      </ComposerPrimitive.Send>
    </div>
  </ComposerPrimitive.Root>
);

export const Thread = () => (
  <ThreadPrimitive.Root className="flex flex-col h-screen max-w-3xl mx-auto bg-white">
    <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
      <ThreadPrimitive.Empty>
        <div className="text-sm text-neutral-400 mt-32 text-center px-4">
          Ask anything about your email.
        </div>
      </ThreadPrimitive.Empty>

      <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
    </ThreadPrimitive.Viewport>

    <Composer />
  </ThreadPrimitive.Root>
);
