"use client";

import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";

import { MarkdownText } from "./MarkdownText";
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

const AssistantMessage = () => (
  <MessagePrimitive.Root className="my-4 px-4">
    <div className="max-w-[95%]">
      <MessagePrimitive.Parts
        components={{
          Text: TextPart,
          tools: { Fallback: ToolCallUI },
        }}
      />
    </div>
  </MessagePrimitive.Root>
);

const Composer = () => (
  <ComposerPrimitive.Root className="border-t bg-white px-4 py-3 flex gap-2">
    <ComposerPrimitive.Input
      placeholder="Ask about your email…"
      className="flex-1 rounded-lg border border-neutral-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
      rows={1}
      autoFocus
    />
    <ComposerPrimitive.Send
      aria-label="Send"
      className="rounded-lg bg-blue-600 text-white px-4 py-2 disabled:opacity-50 flex items-center justify-center"
    >
      {SEND_ICON}
    </ComposerPrimitive.Send>
  </ComposerPrimitive.Root>
);

export const Thread = () => (
  <ThreadPrimitive.Root className="flex flex-col h-screen max-w-3xl mx-auto bg-white">
    <header className="px-4 py-3 border-b">
      <h1 className="text-lg font-semibold">Ask your Gmail</h1>
      <p className="text-xs text-neutral-500">
        Grounded in your local Gmail archive. Click a citation to open the thread.
      </p>
    </header>

    <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
      <ThreadPrimitive.Empty>
        <div className="text-sm text-neutral-500 mt-16 text-center px-4">
          Try: <span className="italic">&ldquo;What did we decide about the roof?&rdquo;</span> or
          <span className="italic"> &ldquo;Newest email from my bank.&rdquo;</span>
        </div>
      </ThreadPrimitive.Empty>

      <ThreadPrimitive.Messages
        components={{
          UserMessage,
          AssistantMessage,
        }}
      />
    </ThreadPrimitive.Viewport>

    <Composer />
  </ThreadPrimitive.Root>
);
