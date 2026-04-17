"use client";

import { useAssistantRuntime } from "@assistant-ui/react";
import { type ReactNode, useEffect, useState } from "react";
import {
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";

import { DebugIdBadge } from "./DebugIdBadge";
import { MarkdownText } from "./MarkdownText";
import { ReasoningPart } from "./ReasoningPart";
import { ToolCallUI } from "./ToolCallUI";

const TextPart = ({ text }: { text: string }) => <MarkdownText text={text} />;

const UserMessage = () => (
  <MessagePrimitive.Root className="flex justify-end my-3 px-3">
    <div className="max-w-[85%] bg-blue-600 text-white rounded-2xl rounded-br-md px-3 py-1.5 text-sm whitespace-pre-wrap">
      <MessagePrimitive.Parts components={{ Text: ({ text }) => <>{text}</> }} />
    </div>
  </MessagePrimitive.Root>
);

const AssistantMessage = () => (
  <MessagePrimitive.Root className="my-3 px-3">
    <div className="max-w-[95%]">
      <MessagePrimitive.Parts
        components={{
          Text: TextPart,
          Reasoning: ReasoningPart,
          tools: { Fallback: ToolCallUI },
          data: {
            by_name: {
              "debug-id": ({ data }: { data: { id: string; log_path: string } }) => (
                <DebugIdBadge data={data} />
              ),
            },
          },
        }}
      />
    </div>
  </MessagePrimitive.Root>
);

// Exposes the assistant runtime to the parent via a callback — the
// battle page needs both runtimes at the top level so it can send the
// same message to both.
const RuntimeBridge = ({ onReady }: { onReady: (r: ReturnType<typeof useAssistantRuntime>) => void }) => {
  const runtime = useAssistantRuntime();
  useEffect(() => {
    onReady(runtime);
  }, [runtime, onReady]);
  return null;
};

type Props = {
  label: ReactNode;
  onRuntimeReady: (r: ReturnType<typeof useAssistantRuntime>) => void;
};

export const BattlePanel = ({ label, onRuntimeReady }: Props) => {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <div className="flex flex-col h-full border border-neutral-200 rounded-lg bg-white overflow-hidden">
      <div className="px-3 py-2 border-b text-xs font-mono text-neutral-500 bg-neutral-50">
        {label}
      </div>
      <ThreadPrimitive.Root className="flex flex-col flex-1 overflow-hidden">
        <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
          <ThreadPrimitive.Empty>
            <div className="text-xs text-neutral-400 mt-10 text-center px-3">
              waiting for question…
            </div>
          </ThreadPrimitive.Empty>
          <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
      {mounted && <RuntimeBridge onReady={onRuntimeReady} />}
    </div>
  );
};
