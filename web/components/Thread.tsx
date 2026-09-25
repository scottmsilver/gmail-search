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
  <MessagePrimitive.Root className="flex justify-end my-4 px-4 sm:px-6 md:px-8">
    <div className="max-w-[85%] sm:max-w-[70%] bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-2 whitespace-pre-wrap">
      <MessagePrimitive.Parts components={{ Text: ({ text }) => <>{text}</> }} />
    </div>
  </MessagePrimitive.Root>
);

type DebugIdData = { id: string; log_path: string };
type CitationWarningData = { broken: string[]; message: string };
type TurnCostData = {
  model: string;
  input_tokens: number;
  output_tokens: number;
  usd: number;
  // Absent on older persisted turns (before this field existed) — treat
  // as "assume cost is known" so old conversations still render their
  // $ figure instead of flipping to "cost unknown" retroactively.
  cost_known?: boolean;
  elapsed_ms?: number | null;
};

const DebugIdPart = ({ data }: { data: DebugIdData }) => <DebugIdBadge data={data} />;
const CitationWarningPart = ({ data }: { data: CitationWarningData }) => (
  <div className="my-2 px-3 py-2 rounded border border-amber-300 bg-amber-50 text-xs text-amber-800">
    ⚠ {data.message} ({data.broken.join(", ")})
  </div>
);

// Short compact-thousands formatter. 1,243 → "1.2K", 2_400_000 → "2.4M",
// keeps token counts readable in a one-line footer.
const compactNum = (n: number): string => {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}K`;
  return `${(n / 1_000_000).toFixed(1)}M`;
};

// USD formatter that shows enough precision to see sub-cent turns.
// $0.0004 is more useful than rounding to "$0.00" when the model is
// Flash Lite.
const fmtUsd = (usd: number): string => {
  if (usd <= 0) return "$0";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(3)}`;
};

// mm:ss above a minute so a long deep-analysis turn doesn't read as
// "83s"; sub-minute turns (the common case) get one decimal so two
// quick turns are still distinguishable.
export const fmtElapsed = (ms: number): string => {
  // elapsed_ms is server-computed (DB clock, always >= 0) in every real
  // caller, but `data-turn-cost` payloads round-trip through JSON from
  // several code paths — guard rather than render "NaNs" or "-3s" if one
  // ever sends something malformed.
  if (!Number.isFinite(ms) || ms < 0) return "";
  // Round to whole seconds FIRST, then branch on that rounded value —
  // rounding the remainder after splitting off whole minutes can carry
  // a value like 119.6s to "1m 60s" (59.6 rounds up to 60 inside the
  // "seconds within the minute" slot, which should have carried into
  // the minutes place instead).
  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) {
    const s = ms / 1000;
    return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  }
  const m = Math.floor(totalSeconds / 60);
  const rem = totalSeconds - m * 60;
  return `${m}m ${rem}s`;
};

const TurnCostPart = ({ data }: { data: TurnCostData }) => {
  // Absent (older persisted turns) reads as "known" — see the field
  // comment on TurnCostData. Only an explicit `false` (the bounded
  // public runtime, which never tracks per-call spend) flips this.
  const costKnown = data.cost_known !== false;
  const timeStr = typeof data.elapsed_ms === "number" ? fmtElapsed(data.elapsed_ms) : "";
  const costStr = costKnown
    ? `${fmtUsd(data.usd)} · ${compactNum(data.input_tokens)} in · ${compactNum(data.output_tokens)} out`
    : "cost unknown";
  const segments = [timeStr, costStr].filter(Boolean);
  if (segments.length === 0) return null;
  return (
    <div
      className="mt-2 text-[11px] text-neutral-400 select-none"
      title={
        costKnown
          ? `${data.model ?? ""} · ${data.input_tokens.toLocaleString()} in / ${data.output_tokens.toLocaleString()} out`
          : "This runtime doesn't track per-call cost"
      }
    >
      {segments.join(" · ")}
    </div>
  );
};

// Render nothing for tool/reasoning parts inline — AssistantWork collects
// them into a single disclosure above the text.
const HiddenPart = () => null;

const AssistantMessage = () => (
  <MessagePrimitive.Root className="my-4 px-4 sm:px-6 md:px-8">
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
            "turn-cost": TurnCostPart as never,
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

type StopControls = {onStop?: () => Promise<void>; stopping?: boolean; stopError?: string | null};

const Composer = ({onStop, stopping, stopError}: StopControls) => (
  // `pb-[calc(...+env(safe-area-inset-bottom))]`: layout.tsx's `viewportFit:
  // "cover"` lets the page draw under the iPhone home-indicator area, so the
  // composer needs its own bottom inset or its controls sit behind it.
  // `env()` is 0 on devices without a safe area, so this is a no-op there.
  <ComposerPrimitive.Root className="px-4 sm:px-6 md:px-8 pb-[calc(0.75rem+env(safe-area-inset-bottom))] pt-1 bg-white">
    <CorpusStatus />
    {/* Below sm the model-picker label ("3.1 Flash Lite · high ▾") eats
        ~140px of a 390px viewport and squeezes the input to two cramped
        lines, so it wraps onto its own row underneath. `order-*` puts
        the input first on that wrapped layout and restores the DOM
        order (picker · input · send) from sm up. */}
    <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-2xl border border-neutral-200 bg-neutral-50 focus-within:border-neutral-400 focus-within:bg-white transition-colors px-3 py-1.5 sm:flex-nowrap sm:pl-2 sm:pr-3">
      <div className="order-last w-full sm:order-none sm:w-auto">
        <ModelPicker />
      </div>
      <ComposerPrimitive.Input
        placeholder="Ask about your email…"
        className="order-first min-w-0 flex-1 bg-transparent focus:outline-none resize-none placeholder:text-neutral-400 text-sm leading-6 sm:order-none"
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
      {!onStop && <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel
          aria-label="Stop"
          title="Stop generating"
          className="self-end w-7 h-7 flex items-center justify-center text-neutral-700 hover:text-neutral-900 transition-colors"
        >
          {WORKING_ICON}
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>}
      {onStop && <button type="button" onClick={() => void onStop()} disabled={stopping}
        aria-label={stopping ? "Stopping" : "Stop"} title={stopping ? "Waiting for worker to stop" : "Stop generating"}
        className="self-end w-7 h-7 flex items-center justify-center disabled:opacity-40">
        {WORKING_ICON}
      </button>}
    </div>
    {stopError && <p role="alert" className="text-sm text-red-700 mt-2">{stopError}</p>}
  </ComposerPrimitive.Root>
);

export const Thread = (controls: StopControls = {}) => (
  <ThreadPrimitive.Root className="flex flex-col h-full w-full max-w-5xl mx-auto bg-white">
    <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
      <ThreadPrimitive.Empty>
        <div className="text-sm text-neutral-400 mt-32 text-center px-4">
          Ask anything about your email.
        </div>
      </ThreadPrimitive.Empty>

      <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
    </ThreadPrimitive.Viewport>

    <Composer {...controls} />
  </ThreadPrimitive.Root>
);
