"use client";

import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";

import { AssistantWork } from "./AssistantWork";
import { BattleMessage } from "./BattleMessage";
import { CorpusStatus } from "./CorpusStatus";
import { DebugIdBadge } from "./DebugIdBadge";
import { MarkdownText } from "./MarkdownText";
import { ModelPicker } from "./ModelPicker";

const SEND_ICON = (
  <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 8h10M9 4l4 4-4 4" />
  </svg>
);

// Spinner + stop-square rendered together: the ring conveys "working"
// while the filled square doubles as the click target + affordance
// for "stop". Uses CSS animation so it keeps spinning across rerenders.
const WORKING_ICON = (
  <span className="relative inline-flex h-5 w-5 items-center justify-center">
    <svg
      className="absolute h-5 w-5 animate-spin text-neutral-400"
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path
        d="M4 12a8 8 0 018-8"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
    <span className="h-2 w-2 rounded-sm bg-neutral-900" aria-hidden="true" />
  </span>
);

const TextPart = ({ text }: { text: string }) => <MarkdownText text={text} />;

const UserMessage = () => (
  <MessagePrimitive.Root className="flex justify-end my-4 px-6 md:px-8">
    <div className="max-w-[70%] bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-2 whitespace-pre-wrap">
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

// Render nothing for tool/reasoning parts inline — AssistantWork collects
// them into a single disclosure above the text.
const HiddenPart = () => null;

const AssistantMessage = () => (
  <MessagePrimitive.Root className="my-4 px-6 md:px-8">
    <AssistantWork />
    <MessagePrimitive.Parts
      components={{
        Text: TextPart,
        Reasoning: HiddenPart,
        tools: { Fallback: HiddenPart },
        data: {
          by_name: {
            "debug-id": DebugIdPart as never,
            "citation-warning": CitationWarningPart as never,
            battle: BattleMessage as never,
            // Register `deep-stage` with a Hidden renderer so assistant-ui
            // surfaces the parts through `useMessage()`. AssistantWork
            // picks them up there and renders them inside its disclosure.
            // Without this explicit registration the parts never reach the
            // hook — assistant-ui filters data-* parts to the ones named
            // here.
            "deep-stage": HiddenPart as never,
          },
        },
      }}
    />
  </MessagePrimitive.Root>
);

const Composer = () => (
  <ComposerPrimitive.Root className="px-6 md:px-8 pb-3 pt-1 bg-white">
    <CorpusStatus />
    <div className="mt-1 flex items-center gap-2 rounded-2xl border border-neutral-200 bg-neutral-50 focus-within:border-neutral-400 focus-within:bg-white transition-colors pl-2 pr-3 py-1.5">
      <ModelPicker />
      <ComposerPrimitive.Input
        placeholder="Ask about your email…"
        className="flex-1 bg-transparent focus:outline-none resize-none placeholder:text-neutral-400 text-sm leading-6"
        rows={1}
        autoFocus
      />
      {/* assistant-ui's ComposerPrimitive.Cancel renders in every state,
          so to swap icons we gate each branch on ThreadPrimitive.If —
          `running` for the stop + spinner, its inverse for send. */}
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send
          aria-label="Send"
          className="self-end w-7 h-7 flex items-center justify-center text-neutral-500 hover:text-neutral-900 disabled:opacity-30 disabled:hover:text-neutral-500 transition-colors"
        >
          {SEND_ICON}
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel
          aria-label="Stop"
          title="Stop generating"
          className="self-end w-7 h-7 flex items-center justify-center text-neutral-700 hover:text-neutral-900 transition-colors"
        >
          {WORKING_ICON}
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>
    </div>
  </ComposerPrimitive.Root>
);

export const Thread = () => (
  <ThreadPrimitive.Root className="flex flex-col h-full w-full max-w-5xl mx-auto bg-white">
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
