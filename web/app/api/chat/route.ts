import { createGoogleGenerativeAI } from "@ai-sdk/google";
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  stepCountIs,
  streamText,
  type ModelMessage,
  type UIMessage,
} from "ai";
import { NextRequest } from "next/server";

import {
  AGENT_MODEL,
  DEFAULT_THINKING,
  geminiApiKey,
  isValidModel,
  isValidThinking,
  type ThinkingLevel,
} from "@/lib/config";
import { lookupThreadByCiteRef } from "@/lib/backend";
import { ChatLogger } from "@/lib/chatLog";
import { collectKnownRefs, findBrokenRefs } from "@/lib/citationCheck";
import { buildSystemPrompt } from "@/lib/systemPrompt";
import { buildTools } from "@/lib/tools";

export const runtime = "nodejs";
export const maxDuration = 180;

const MAX_TOOL_STEPS = 6;
const MAX_VALIDATION_RETRIES = 2;

const buildCorrectionMessage = (broken: string[]): string =>
  `Your previous answer cited refs that are not in any tool result: ${broken
    .map((b) => `[ref:${b}]`)
    .join(", ")}.

For EACH broken ref, do ONE of these:
1. If you remember which thread you meant, call validate_reference with the prefix you have to recover the correct cite_ref, then re-emit the answer with the validated cite_ref.
2. If you cannot validate it, REMOVE the citation entirely from your answer. Do not invent or reuse a different cite_ref.

Now re-write your previous answer end-to-end (don't apologize, don't explain — just produce the corrected answer).`;

// Drop assistant text chunks while letting tool / reasoning / data chunks
// pass through. We use this on intermediate (un-validated) attempts so
// only the final, validated answer text reaches the client.
type AnyChunk = { type?: string } & Record<string, unknown>;
const suppressTextChunks = <T extends AnyChunk>(): TransformStream<T, T> => {
  return new TransformStream<T, T>({
    transform(chunk, controller) {
      const type = chunk?.type ?? "";
      if (type === "text-start" || type === "text-delta" || type === "text-end") return;
      controller.enqueue(chunk);
    },
  });
};

const lastUserText = (messages: UIMessage[]): string => {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "user") continue;
    for (const p of m.parts) {
      if (p.type === "text") return p.text;
    }
  }
  return "";
};

export async function POST(req: NextRequest) {
  const body = (await req.json()) as {
    messages?: UIMessage[];
    model?: string;
    thinkingLevel?: string;
  };
  const messages = body.messages;

  if (!Array.isArray(messages) || messages.length === 0) {
    return new Response(JSON.stringify({ error: "messages required" }), { status: 400 });
  }

  // Per-request overrides from the UI picker; fall back to defaults.
  const model = isValidModel(body.model) ? body.model : AGENT_MODEL;
  const thinkingLevel: ThinkingLevel = isValidThinking(body.thinkingLevel)
    ? body.thinkingLevel
    : DEFAULT_THINKING;

  const logger = new ChatLogger();
  const question = lastUserText(messages);
  await logger.log("request", {
    id: logger.id,
    question,
    message_count: messages.length,
    model,
    thinking_level: thinkingLevel,
  });
  console.log(
    `[chat ${logger.id}] start model=${model} thinking=${thinkingLevel} q=${JSON.stringify(question.slice(0, 80))}`,
  );

  const google = createGoogleGenerativeAI({ apiKey: geminiApiKey() });
  const tools = buildTools();
  const system = buildSystemPrompt(tools);
  const initialMessages = await convertToModelMessages(messages);

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      // Surface the request ID to the client immediately so the UI can
      // show a copyable badge while the response streams.
      writer.write({
        type: "data-debug-id",
        id: `dbg-${logger.id}`,
        data: { id: logger.id, log_path: logger.path },
      });

      const conversation: ModelMessage[] = [...initialMessages];
      const knownRefs = new Set<string>();

      for (let attempt = 0; attempt <= MAX_VALIDATION_RETRIES; attempt++) {
        await logger.log("attempt", { attempt: attempt + 1 });
        const isFinalAttempt = attempt === MAX_VALIDATION_RETRIES;

        const result = streamText({
          model: google(model),
          system,
          messages: conversation,
          tools,
          stopWhen: stepCountIs(MAX_TOOL_STEPS),
          providerOptions: {
            google: {
              // NOTE: On Gemini 3.1 Flash Lite, "medium" and "high" think
              // more but do NOT stream thought summaries — the Thoughts
              // UI panel goes blank. "low" is the only level that emits
              // visible reasoning. Trade reasoning depth vs transparency.
              thinkingConfig: { thinkingLevel, includeThoughts: true },
            },
          },
          onError: ({ error }) => {
            const msg = error instanceof Error ? error.message : String(error);
            console.error(`[chat ${logger.id}] streamText error:`, error);
            void logger.log("error", { source: "streamText", message: msg });
          },
        });

        // Stream tool calls + reasoning live so the UI shows progress, but
        // hold back the assistant TEXT until validation passes — otherwise
        // a broken-citation answer flashes to the user before we retry.
        // (Codex #3 — validation loop streamed every attempt.)
        const uiStream = result.toUIMessageStream({ sendReasoning: true });
        if (isFinalAttempt) {
          writer.merge(uiStream);
        } else {
          writer.merge(uiStream.pipeThrough(suppressTextChunks()));
        }

        const [finalText, steps, reasoning] = await Promise.all([
          result.text,
          result.steps,
          result.reasoningText,
        ]);

        if (reasoning) {
          await logger.log("reasoning", { attempt: attempt + 1, text: reasoning });
        }

        for (const step of steps) {
          for (const tc of step.toolCalls ?? []) {
            await logger.log("tool_call", {
              attempt: attempt + 1,
              name: tc.toolName,
              args: tc.input,
            });
          }
          for (const tr of step.toolResults ?? []) {
            collectKnownRefs(tr.output, knownRefs);
            await logger.log("tool_result", {
              attempt: attempt + 1,
              name: tr.toolName,
              output: tr.output,
            });
          }
        }

        // Parallelize the lookup fallback. findBrokenRefs walks each ref;
        // if many are unknown we'd otherwise sequentially round-trip Python.
        const lookupCache = new Map<string, Promise<string | null>>();
        const cachedLookup = (ref: string): Promise<string | null> => {
          const hit = lookupCache.get(ref);
          if (hit) return hit;
          const p = lookupThreadByCiteRef(ref).then((res) => {
            if (res.ok) {
              knownRefs.add(res.hit.thread_id);
              return res.hit.thread_id;
            }
            return null;
          });
          lookupCache.set(ref, p);
          return p;
        };
        const { broken, total } = await findBrokenRefs(finalText, knownRefs, cachedLookup);
        await logger.log("answer", {
          attempt: attempt + 1,
          text: finalText,
          citations: total,
          broken,
        });
        await logger.log("validation", {
          attempt: attempt + 1,
          known_refs: [...knownRefs].length,
          citations: total,
          broken,
        });
        console.log(
          `[chat ${logger.id}] attempt=${attempt + 1} citations=${total} broken=${broken.length}` +
            (broken.length ? ` (${broken.join(", ")})` : ""),
        );

        if (broken.length === 0) {
          // For non-final attempts whose text we suppressed, replay the
          // validated answer as one text part so the UI sees it.
          if (!isFinalAttempt) {
            const id = `valid-text-${attempt}`;
            writer.write({ type: "text-start", id });
            writer.write({ type: "text-delta", id, delta: finalText });
            writer.write({ type: "text-end", id });
          }
          await logger.log("done", { attempts: attempt + 1, ms: logger.elapsedMs() });
          return;
        }
        if (attempt === MAX_VALIDATION_RETRIES) {
          writer.write({
            type: "data-citation-warning",
            id: `cw-${attempt}`,
            data: {
              broken,
              message: `Could not validate ${broken.length} citation(s) after ${
                MAX_VALIDATION_RETRIES + 1
              } attempts.`,
            },
          });
          await logger.log("done", {
            attempts: attempt + 1,
            ms: logger.elapsedMs(),
            unresolved_broken: broken,
          });
          return;
        }

        conversation.push({ role: "assistant", content: finalText });
        conversation.push({ role: "user", content: buildCorrectionMessage(broken) });
      }
    },
    onError: (error) => {
      const msg = error instanceof Error ? error.message : String(error);
      console.error(`[chat ${logger.id}] stream error:`, error);
      void logger.log("error", { source: "stream", message: msg });
      return msg;
    },
  });

  return createUIMessageStreamResponse({ stream });
}
