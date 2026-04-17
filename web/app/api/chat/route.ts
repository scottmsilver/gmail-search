import { createGoogleGenerativeAI } from "@ai-sdk/google";
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  generateText,
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
  pythonApiUrl,
  type ThinkingLevel,
} from "@/lib/config";
import { lookupThreadByCiteRef } from "@/lib/backend";
import { pickTwoRandomVariants } from "@/lib/battleVariants";
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

// Auto-title: trim the first user message to ~60 chars, single line.
const deriveTitle = (messages: UIMessage[]): string => {
  for (const m of messages) {
    if (m.role !== "user") continue;
    for (const p of m.parts) {
      if (p.type === "text") {
        const one = p.text.replace(/\s+/g, " ").trim();
        return one.length > 60 ? one.slice(0, 57) + "…" : one || "New chat";
      }
    }
  }
  return "New chat";
};

// Persist the conversation (user messages + newly-finished assistant
// turn) to Python. Fire-and-forget — a save failure should never break
// the chat response. Called after the stream has produced its answer.
const saveConversation = async (
  conversationId: string,
  messages: Array<{ role: string; parts: unknown[] }>,
  title: string | null,
): Promise<void> => {
  try {
    const res = await fetch(
      `${pythonApiUrl()}/api/conversations/${encodeURIComponent(conversationId)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, messages }),
      },
    );
    if (!res.ok) {
      console.error(`[chat] save conversation ${conversationId} failed: ${res.status}`);
    }
  } catch (err) {
    console.error(`[chat] save conversation ${conversationId} threw:`, err);
  }
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

// Only Gemini 3.x supports thinkingConfig. 2.x rejects it with "Thinking
// level is not supported for this model." Build provider options per-model.
const googleProviderOptions = (model: string, thinkingLevel: ThinkingLevel, includeThoughts: boolean) => {
  if (!model.startsWith("gemini-3")) return undefined;
  return { google: { thinkingConfig: { thinkingLevel, includeThoughts } } };
};

// Run one variant end-to-end (no validation loop, no streaming to client)
// and return the final text + a compact tool log. Used by battle mode.
const runVariantOnce = async (
  google: ReturnType<typeof createGoogleGenerativeAI>,
  system: string,
  tools: ReturnType<typeof buildTools>,
  conversation: ModelMessage[],
  model: string,
  thinkingLevel: ThinkingLevel,
): Promise<{
  text: string;
  tools: Array<{ name: string; args: unknown; output: unknown }>;
  error?: string;
}> => {
  try {
    const result = await generateText({
      model: google(model),
      system,
      messages: conversation,
      tools,
      stopWhen: stepCountIs(MAX_TOOL_STEPS),
      providerOptions: googleProviderOptions(model, thinkingLevel, false),
    });
    const toolLog: Array<{ name: string; args: unknown; output: unknown }> = [];
    for (const step of result.steps) {
      for (const tr of step.toolResults ?? []) {
        toolLog.push({ name: tr.toolName, args: tr.input, output: tr.output });
      }
    }
    // Some models (notably 2.5 Flash Lite) call a tool and then bail
    // without emitting any text. Treat that as a failure so the UI
    // doesn't show an empty panel — the user deserves to see WHY.
    if (!result.text || result.text.trim().length === 0) {
      const finish = result.finishReason ?? "unknown";
      return {
        text: "",
        tools: toolLog,
        error: `Model returned no text (finish reason: ${finish}, ${toolLog.length} tool call(s)). This model may not be good at this kind of question — try a different variant.`,
      };
    }
    return { text: result.text, tools: toolLog };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[chat] battle variant ${model}/${thinkingLevel} failed:`, err);
    return { text: "", tools: [], error: msg };
  }
};

export async function POST(req: NextRequest) {
  const body = (await req.json()) as {
    messages?: UIMessage[];
    model?: string;
    thinkingLevel?: string;
    battle?: boolean;
    conversation_id?: string;
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
  const conversationId = body.conversation_id || null;
  await logger.log("request", {
    id: logger.id,
    question,
    message_count: messages.length,
    model,
    thinking_level: thinkingLevel,
    conversation_id: conversationId,
  });
  console.log(
    `[chat ${logger.id}] start model=${model} thinking=${thinkingLevel} q=${JSON.stringify(question.slice(0, 80))}`,
  );

  const google = createGoogleGenerativeAI({ apiKey: geminiApiKey() });
  const tools = buildTools();
  const system = buildSystemPrompt(tools);
  const initialMessages = await convertToModelMessages(messages);

  // ── Battle mode ─────────────────────────────────────────────────
  // Run two random variants in parallel and emit a single data-battle
  // part. No validation loop, no per-variant streaming — the UI shows
  // both complete answers side-by-side and lets the user vote.
  if (body.battle) {
    const [variantA, variantB] = pickTwoRandomVariants();
    // Log the inbound conversation so we can diagnose "did we actually
    // send context?" complaints after the fact.
    await logger.log("battle_start", {
      variant_a: variantA,
      variant_b: variantB,
      context_summary: initialMessages.map((m) => ({
        role: m.role,
        preview:
          typeof m.content === "string"
            ? m.content.slice(0, 200)
            : Array.isArray(m.content)
              ? `<${m.content.length} parts>`
              : "<complex>",
      })),
    });
    console.log(
      `[chat ${logger.id}] battle: ${variantA.model}/${variantA.thinkingLevel} vs ${variantB.model}/${variantB.thinkingLevel}`,
    );

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        writer.write({
          type: "data-debug-id",
          id: `dbg-${logger.id}`,
          data: { id: logger.id, log_path: logger.path },
        });

        const battleId = `battle-${logger.id}`;
        const emit = (partial: Record<string, unknown>) => {
          writer.write({
            type: "data-battle",
            id: battleId,
            data: {
              request_id: logger.id,
              question,
              variant_a: variantA,
              variant_b: variantB,
              ...partial,
            },
          });
        };

        // Show the battle panel immediately with both sides pending. The
        // client renders spinners while each variant is working.
        emit({
          running_a: true,
          running_b: true,
          answer_a: "",
          answer_b: "",
          tools_a: [],
          tools_b: [],
        });

        // Launch both variants; flip the running flag per side as each
        // finishes so the user gets incremental feedback.
        let finalA: Awaited<ReturnType<typeof runVariantOnce>> | null = null;
        let finalB: Awaited<ReturnType<typeof runVariantOnce>> | null = null;

        const finishSide = (side: "a" | "b", r: typeof finalA) => {
          if (!r) return;
          if (side === "a") finalA = r;
          else finalB = r;
          emit({
            running_a: finalA === null,
            running_b: finalB === null,
            answer_a: finalA ? (finalA.error ? `⚠ ${finalA.error}` : finalA.text) : "",
            answer_b: finalB ? (finalB.error ? `⚠ ${finalB.error}` : finalB.text) : "",
            tools_a: finalA?.tools ?? [],
            tools_b: finalB?.tools ?? [],
          });
        };

        const pa = runVariantOnce(
          google,
          system,
          tools,
          initialMessages,
          variantA.model,
          variantA.thinkingLevel,
        ).then((r) => finishSide("a", r));
        const pb = runVariantOnce(
          google,
          system,
          tools,
          initialMessages,
          variantB.model,
          variantB.thinkingLevel,
        ).then((r) => finishSide("b", r));
        await Promise.all([pa, pb]);

        // Record full answer text + tool calls per side so we can audit
        // quality later. Base64 is stripped by the logger.
        const toLogRecord = (r: typeof finalA) =>
          r
            ? {
                text: (r as { text: string }).text,
                error: (r as { error?: string }).error,
                tools: (r as { tools: Array<{ name: string; args: unknown; output: unknown }> }).tools,
              }
            : { text: "", error: "no result" };
        await logger.log("battle_done", {
          variant_a: variantA,
          variant_b: variantB,
          a: toLogRecord(finalA),
          b: toLogRecord(finalB),
          ms: logger.elapsedMs(),
        });

        // Persist: prior messages (as sent) + the final battle record
        // as one assistant turn with role "assistant" and a data-battle part.
        if (conversationId) {
          const persisted: Array<{ role: string; parts: unknown[] }> = messages.map((m) => ({
            role: m.role,
            parts: m.parts as unknown[],
          }));
          persisted.push({
            role: "assistant",
            parts: [
              {
                type: "data-battle",
                id: battleId,
                data: {
                  request_id: logger.id,
                  question,
                  variant_a: variantA,
                  variant_b: variantB,
                  answer_a: finalA ? ((finalA as { error?: string }).error ? `⚠ ${(finalA as { error: string }).error}` : (finalA as { text: string }).text) : "",
                  answer_b: finalB ? ((finalB as { error?: string }).error ? `⚠ ${(finalB as { error: string }).error}` : (finalB as { text: string }).text) : "",
                  tools_a: finalA ? (finalA as { tools: unknown[] }).tools : [],
                  tools_b: finalB ? (finalB as { tools: unknown[] }).tools : [],
                },
              },
            ],
          });
          await saveConversation(conversationId, persisted, deriveTitle(messages));
        }
      },
      onError: (error) => {
        const msg = error instanceof Error ? error.message : String(error);
        console.error(`[chat ${logger.id}] battle error:`, error);
        return msg;
      },
    });
    return createUIMessageStreamResponse({ stream });
  }

  // ── Normal mode (validation loop, streaming) ────────────────────
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
          // Thinking config only applies to Gemini 3.x; 2.x rejects it.
          // NOTE: On Gemini 3.1 Flash Lite, "medium" and "high" still think
          // more but do NOT stream thought summaries, so the Thoughts
          // panel stays empty above "low".
          providerOptions: googleProviderOptions(model, thinkingLevel, true),
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
          if (conversationId) {
            const persisted: Array<{ role: string; parts: unknown[] }> = messages.map((m) => ({
              role: m.role,
              parts: m.parts as unknown[],
            }));
            persisted.push({
              role: "assistant",
              parts: [{ type: "text", text: finalText }],
            });
            await saveConversation(conversationId, persisted, deriveTitle(messages));
          }
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
