"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { extractThreadHints } from "@/lib/extractThreadHints";
import { linkifyRefs, REF_PREFIX } from "@/lib/linkifyRefs";
import { variantLabel, type BattleVariant } from "@/lib/battleVariants";

import { CitationChip } from "./CitationChip";
import { useThreadDrawer } from "./ThreadDrawerContext";

type BattleData = {
  request_id: string;
  question: string;
  variant_a: BattleVariant;
  variant_b: BattleVariant;
  answer_a: string;
  answer_b: string;
  tools_a: Array<{ name: string; args: unknown; output: unknown }>;
  tools_b: Array<{ name: string; args: unknown; output: unknown }>;
};

type Vote = "a" | "b" | "tie" | "both_bad";

const BattleSide = ({
  label,
  text,
  tools,
  side,
  won,
}: {
  label: string;
  text: string;
  tools: BattleData["tools_a"];
  side: "a" | "b";
  won: boolean | null;
}) => {
  const { setOpenThreadId } = useThreadDrawer();
  // Walk tool outputs just like the normal renderer so [ref:] chips resolve.
  const hints = extractThreadHints(
    tools.map((t) => ({ type: "tool-call" as const, result: t.output })),
  );
  const knownIds = hints.map((h) => h.thread_id);

  const colorCls =
    won === true
      ? "border-emerald-300 bg-emerald-50/40"
      : won === false
      ? "border-neutral-200 bg-neutral-50/40 opacity-70"
      : "border-neutral-200";

  return (
    <div className={`rounded-lg border ${colorCls} p-3 flex flex-col min-w-0`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono text-neutral-500">{label}</span>
        <span className="text-[10px] text-neutral-400">
          {tools.length} tool call{tools.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="prose prose-sm max-w-none prose-p:my-1.5 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0 prose-headings:my-2 prose-pre:my-2">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          urlTransform={(url) => url}
          components={{
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
          }}
        >
          {linkifyRefs(text, knownIds)}
        </ReactMarkdown>
      </div>
    </div>
  );
};

export const BattleMessage = ({ data }: { data: BattleData }) => {
  const [vote, setVote] = useState<Vote | null>(null);
  const [voting, setVoting] = useState(false);

  const submitVote = async (winner: Vote) => {
    if (vote || voting) return;
    setVoting(true);
    try {
      await fetch("/api/battle/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: data.question,
          variant_a: data.variant_a,
          variant_b: data.variant_b,
          winner,
          request_id_a: data.request_id,
          request_id_b: data.request_id,
        }),
      });
      setVote(winner);
    } catch (e) {
      console.error(e);
    } finally {
      setVoting(false);
    }
  };

  const revealed = vote !== null;
  const labelA = revealed ? variantLabel(data.variant_a) : "A";
  const labelB = revealed ? variantLabel(data.variant_b) : "B";
  const wonA = vote === "a" ? true : vote === "b" ? false : null;
  const wonB = vote === "b" ? true : vote === "a" ? false : null;

  const btn = (v: Vote, label: string, hoverCls = "hover:bg-blue-50") => (
    <button
      key={v}
      type="button"
      disabled={revealed || voting}
      onClick={() => submitVote(v)}
      className={`rounded-md border border-neutral-300 px-3 py-1 text-xs ${hoverCls} disabled:opacity-40 disabled:cursor-not-allowed`}
    >
      {label}
    </button>
  );

  return (
    <div className="my-3">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <BattleSide label={labelA} text={data.answer_a} tools={data.tools_a} side="a" won={wonA} />
        <BattleSide label={labelB} text={data.answer_b} tools={data.tools_b} side="b" won={wonB} />
      </div>
      <div className="mt-2 flex items-center gap-2 flex-wrap">
        {revealed ? (
          <span className="text-xs text-emerald-700">
            ✓ recorded:{" "}
            {vote === "tie"
              ? "tie"
              : vote === "both_bad"
              ? "both bad"
              : vote === "a"
              ? `A wins — ${variantLabel(data.variant_a)}`
              : `B wins — ${variantLabel(data.variant_b)}`}
          </span>
        ) : (
          <>
            <span className="text-xs text-neutral-500">which is better?</span>
            {btn("a", "A wins")}
            {btn("b", "B wins")}
            {btn("tie", "Same", "hover:bg-neutral-50")}
            {btn("both_bad", "Both bad", "hover:bg-red-50")}
          </>
        )}
      </div>
    </div>
  );
};
