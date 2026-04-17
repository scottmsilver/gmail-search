"use client";

import Link from "next/link";
import { useCallback, useMemo, useRef, useState } from "react";

import { AssistantRuntimeProvider, useAssistantRuntime } from "@assistant-ui/react";
import { AssistantChatTransport, useChatRuntime } from "@assistant-ui/react-ai-sdk";

import { BattlePanel } from "@/components/BattlePanel";
import { ThreadDrawer } from "@/components/ThreadDrawer";
import { ThreadDrawerProvider, useThreadDrawer } from "@/components/ThreadDrawerContext";
import {
  BATTLE_VARIANTS,
  pickTwoRandomVariants,
  variantLabel,
  type BattleVariant,
} from "@/lib/battleVariants";

const PYTHON_UI_URL = process.env.NEXT_PUBLIC_PYTHON_UI_URL ?? "http://127.0.0.1:8080";

const DrawerHost = () => {
  const { openThreadId, setOpenThreadId } = useThreadDrawer();
  return (
    <ThreadDrawer
      threadId={openThreadId}
      onClose={() => setOpenThreadId(null)}
      pythonBaseUrl={PYTHON_UI_URL}
    />
  );
};

type Runtime = ReturnType<typeof useAssistantRuntime>;

const useVariantRuntime = (variant: BattleVariant) => {
  const transport = useMemo(
    () =>
      new AssistantChatTransport({
        api: "/api/chat",
        body: () => ({ model: variant.model, thinkingLevel: variant.thinkingLevel }),
      }),
    [variant],
  );
  return useChatRuntime({ transport });
};

type Winner = "a" | "b" | "tie" | "both_bad";

export default function BattlePage() {
  const [pair, setPair] = useState<[BattleVariant, BattleVariant]>(() => pickTwoRandomVariants());
  const [variantA, variantB] = pair;
  const [question, setQuestion] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const [revealed, setRevealed] = useState(false);
  const [voting, setVoting] = useState(false);
  const [voteMsg, setVoteMsg] = useState<string | null>(null);

  const runtimeA = useVariantRuntime(variantA);
  const runtimeB = useVariantRuntime(variantB);
  const runtimeARef = useRef<Runtime | null>(null);
  const runtimeBRef = useRef<Runtime | null>(null);

  const submit = useCallback(() => {
    const text = question.trim();
    if (!text) return;
    setLastQuestion(text);
    setRevealed(false);
    setVoteMsg(null);
    runtimeARef.current?.thread.append(text);
    runtimeBRef.current?.thread.append(text);
    setQuestion("");
  }, [question]);

  const vote = useCallback(
    async (winner: Winner) => {
      if (!lastQuestion || voting) return;
      setVoting(true);
      try {
        const res = await fetch("/api/battle/vote", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: lastQuestion,
            variant_a: variantA,
            variant_b: variantB,
            winner,
          }),
        });
        if (!res.ok) throw new Error(`vote failed: ${res.status}`);
        setRevealed(true);
        setVoteMsg(`vote recorded: ${winner === "tie" ? "tie" : winner === "both_bad" ? "both bad" : winner.toUpperCase() + " wins"}`);
      } catch (err) {
        setVoteMsg(`error: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setVoting(false);
      }
    },
    [lastQuestion, variantA, variantB, voting],
  );

  const nextPair = useCallback(() => {
    setPair(pickTwoRandomVariants());
    setRevealed(false);
    setVoteMsg(null);
    setLastQuestion("");
    // The old runtimes linger but won't be used. Their threads stay visible
    // above until the user refreshes — acceptable for this MVP.
  }, []);

  const labelFor = (which: "a" | "b", v: BattleVariant) =>
    revealed ? variantLabel(v) : which.toUpperCase();

  return (
    <ThreadDrawerProvider>
      <div className="flex flex-col h-screen max-w-6xl mx-auto bg-white">
        <header className="px-4 py-3 border-b flex items-center gap-4">
          <Link href="/" className="text-sm text-neutral-500 hover:text-neutral-800">← chat</Link>
          <h1 className="text-base font-semibold">Model battle</h1>
          <span className="text-xs text-neutral-400">
            blind A/B · vote to reveal · {BATTLE_VARIANTS.length} variants
          </span>
          <div className="ml-auto">
            <Link href="/battle/stats" className="text-xs text-blue-600 hover:underline">
              leaderboard →
            </Link>
          </div>
        </header>

        <form
          className="px-4 py-3 border-b flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask both models the same question…"
            className="flex-1 rounded-lg border border-neutral-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={!question.trim()}
            className="rounded-lg bg-neutral-900 text-white px-4 py-2 text-sm disabled:opacity-40 hover:bg-neutral-700"
          >
            Ask both
          </button>
          <button
            type="button"
            onClick={nextPair}
            className="rounded-lg border border-neutral-300 text-neutral-600 px-3 py-2 text-sm hover:bg-neutral-50"
            title="pick a new random pair of variants"
          >
            ↻ re-roll
          </button>
        </form>

        <div className="flex-1 grid grid-cols-2 gap-3 px-4 py-3 overflow-hidden">
          <AssistantRuntimeProvider runtime={runtimeA}>
            <BattlePanel
              label={labelFor("a", variantA)}
              onRuntimeReady={(r) => (runtimeARef.current = r)}
            />
          </AssistantRuntimeProvider>
          <AssistantRuntimeProvider runtime={runtimeB}>
            <BattlePanel
              label={labelFor("b", variantB)}
              onRuntimeReady={(r) => (runtimeBRef.current = r)}
            />
          </AssistantRuntimeProvider>
        </div>

        <footer className="border-t px-4 py-3 flex items-center gap-2">
          <span className="text-xs text-neutral-500 mr-2">which was better?</span>
          <button
            type="button"
            disabled={!lastQuestion || revealed || voting}
            onClick={() => vote("a")}
            className="rounded-md border px-3 py-1.5 text-sm hover:bg-blue-50 disabled:opacity-40"
          >
            A wins
          </button>
          <button
            type="button"
            disabled={!lastQuestion || revealed || voting}
            onClick={() => vote("b")}
            className="rounded-md border px-3 py-1.5 text-sm hover:bg-blue-50 disabled:opacity-40"
          >
            B wins
          </button>
          <button
            type="button"
            disabled={!lastQuestion || revealed || voting}
            onClick={() => vote("tie")}
            className="rounded-md border px-3 py-1.5 text-sm hover:bg-neutral-50 disabled:opacity-40"
          >
            Tie
          </button>
          <button
            type="button"
            disabled={!lastQuestion || revealed || voting}
            onClick={() => vote("both_bad")}
            className="rounded-md border px-3 py-1.5 text-sm hover:bg-red-50 disabled:opacity-40"
          >
            Both bad
          </button>
          {voteMsg && <span className="ml-3 text-xs text-emerald-700">{voteMsg}</span>}
        </footer>
        <DrawerHost />
      </div>
    </ThreadDrawerProvider>
  );
}
