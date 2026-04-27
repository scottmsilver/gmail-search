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
import { sanitizeConversation } from "@/lib/sanitizeConversation";
import { estimateCostUsd } from "@/lib/pricing";
import { getSqlSchemaMarkdown } from "@/lib/sqlSchema";
import { buildSystemPrompt } from "@/lib/systemPrompt";
import { buildTools } from "@/lib/tools";

export const runtime = "nodejs";
export const maxDuration = 180;

const MAX_TOOL_STEPS = 15;
const MAX_VALIDATION_RETRIES = 2;

// Map known provider errors to something a user can act on. Gemini's
// 1,048,576-token input cap surfaces as a verbose stack trace otherwise.
const friendlyModelError = (msg: string): string => {
  if (/input token count exceeds the maximum|exceeds the maximum number of tokens/i.test(msg)) {
    return "This variant ran out of context while investigating (over 1M tokens). Try a narrower question, or the other variant may still have an answer.";
  }
  return msg;
};

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

// sanitizeConversation moved to web/lib/sanitizeConversation so it can be
// unit-tested without spinning up a Next.js route.

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

// Compose the full tool catalog: the backend-backed user tools
// (search_emails, sql_query, get_thread, ...) plus Gemini's built-in
// grounded web search. The key MUST be `google_search` — the provider
// tool factory asserts on the name.
const buildAllTools = (google: ReturnType<typeof createGoogleGenerativeAI>) => ({
  ...buildTools(),
  google_search: google.tools.googleSearch({}),
});

// Run one variant end-to-end (no validation loop, no streaming to client)
// and return the final text + a compact tool log. Used by battle mode.
const runVariantOnce = async (
  google: ReturnType<typeof createGoogleGenerativeAI>,
  system: string,
  tools: ReturnType<typeof buildAllTools>,
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
      // Dig for provider-level details: rawFinishReason from the model,
      // warnings, or the last step's raw response body (Gemini error
      // payloads land here when finishReason === "error").
      const lastStep =
        result.steps && result.steps.length > 0
          ? result.steps[result.steps.length - 1]
          : undefined;
      const bodyText = (() => {
        const body = (lastStep?.response as { body?: unknown } | undefined)?.body;
        if (!body) return "";
        if (typeof body === "string") return body;
        try {
          return JSON.stringify(body);
        } catch {
          return String(body);
        }
      })();
      const rawFinish = (result as { rawFinishReason?: string }).rawFinishReason ?? "";
      const warningsText = result.warnings
        ?.map((w) => (w as { message?: string }).message ?? String(w))
        .join("; ");
      const providerHint =
        warningsText || rawFinish || bodyText.slice(0, 240) || "";
      // bodyText can echo email subject lines back from the provider request.
      // Don't dump it in production logs.
      const isProd = process.env.NODE_ENV === "production";
      console.error(
        `[chat] battle variant ${model}/${thinkingLevel} empty finish=${finish} raw=${rawFinish} warnings=${warningsText ?? ""}`,
        ...(isProd ? [] : ["body=", bodyText.slice(0, 500)]),
      );
      // finish_reason === "tool-calls" means the model wanted another
      // tool call but we stopped it at MAX_TOOL_STEPS — that's our cap,
      // not a model failure.
      let error: string;
      if (finish === "tool-calls") {
        error = `Model reached its thinking budget after ${toolLog.length} tool call(s) and may have had more to say.`;
      } else if (finish === "error") {
        error = `Model request errored (${providerHint || "no details from provider"}). Try a different variant.`;
      } else {
        error = `Model returned no text (finish reason: ${finish}, ${toolLog.length} tool call(s)). This model may not be good at this kind of question — try a different variant.`;
      }
      return { text: "", tools: toolLog, error };
    }
    return { text: result.text, tools: toolLog };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[chat] battle variant ${model}/${thinkingLevel} threw:`, err);
    return { text: "", tools: [], error: friendlyModelError(msg) };
  }
};

// ── Deep-mode translator ──────────────────────────────────────────
//
// The Python agent service streams SSE with a small vocabulary:
//   session   — assigns session_id (unused on the chat side)
//   plan      — Planner emitted JSON
//   evidence  — Retriever summary
//   tool_call — retriever called search/query/sql/get_thread
//   analysis  — Analyst summary after its run_code calls
//   skipped   — Analyst skipped (plan had no analysis steps)
//   code_run  — Analyst ran a snippet
//   draft     — Writer draft (may be revised by critic)
//   critique  — Critic JSON verdict
//   revision  — Writer rewrite
//   cost      — per-stage token usage + USD
//   final     — final markdown answer
//   error     — orchestrator raised
//
// We translate each kind into a UIMessageStream part the chat UI
// already renders:
//   - stage events (plan / evidence / analysis / critique / cost)
//     become `data-deep-stage` parts so the Thread can render a
//     "Working (N steps)" disclosure just like regular tool calls.
//   - `final` → text chunk in the assistant bubble (with [ref:],
//     [att:], [art:] citation chips from CitableMarkdown).
//   - `error` → text chunk explaining the failure.

type DeepStreamArgs = {
  question: string;
  conversationId: string | null;
  loggerId: string;
  loggerPath: string;
  // Full inbound messages so we can persist the user turn alongside
  // the assistant's deep-mode reply on conversation save. Without
  // this the user reloads the conversation and sees only the half
  // they typed (or worse, nothing).
  messages: UIMessage[];
  // Picker-selected model — applied to every deep sub-agent
  // (Planner / Retriever / Analyst / Writer / Critic). When omitted,
  // Python falls back to per-stage env vars / built-in defaults.
  model: string;
  // Which deep backend the Python service should run: "adk" (default),
  // "claude_code" (claudebox + orchestrator), or "claude_native"
  // (single-agent claudebox loop). Forwarded as `backend`.
  deepBackend?: "adk" | "claude_code" | "claude_native";
};

const createDeepModeStream = (args: DeepStreamArgs) =>
  createUIMessageStream({
    execute: async ({ writer }) => {
      writer.write({
        type: "data-debug-id",
        id: `dbg-${args.loggerId}`,
        data: { id: args.loggerId, log_path: args.loggerPath },
      });

      // Persist the (user message(s) + assistant text) tuple so a
      // refresh shows the deep-mode answer. Used ONLY by the error/
      // empty-text fallback branches below — for the happy path, the
      // Python service writes a RICH assistant message (with all
      // tool-call blocks reconstructed from agent_events) inside
      // `_real_run`. Letting this client-side path also fire on the
      // happy path would race and clobber the rich row with a text-
      // only one (the bug we hit on 2026-04-27). On error paths
      // there's nothing rich to preserve, so the text-only fallback
      // is the right write.
      const persistAssistantText = async (assistantText: string): Promise<void> => {
        if (!args.conversationId) return;
        const persisted: Array<{ role: string; parts: unknown[] }> = args.messages.map(
          (m) => ({ role: m.role, parts: m.parts as unknown[] }),
        );
        persisted.push({
          role: "assistant",
          parts: [{ type: "text", text: assistantText }],
        });
        await saveConversation(args.conversationId, persisted, deriveTitle(args.messages));
      };

      const upstream = await fetch(`${pythonApiUrl()}/api/agent/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: args.question,
          conversation_id: args.conversationId,
          model: args.model,
          backend: args.deepBackend,
        }),
      });
      if (!upstream.ok || !upstream.body) {
        // Pull as much detail as the upstream gave us — the body
        // usually has the FastAPI error reason. Clip to 500 chars so
        // a giant HTML error page doesn't drown the bubble.
        let detail = "";
        try {
          detail = (await upstream.text()).slice(0, 500);
        } catch {
          /* body read can fail if the connection is already torn; fall
             back to status code only */
        }
        const detailLine = detail.trim() ? ` ${detail.trim()}` : "";
        const errMsg =
          `Deep-mode upstream failed (HTTP ${upstream.status}).${detailLine}`;
        // Surface it as a citation-warning-shaped data part so the UI
        // renders an explicit error block (red/amber border) rather
        // than an italic one-liner that looks like model output. The
        // Thread already has a CitationWarningPart handler.
        writer.write({
          type: "data-citation-warning",
          id: `deep-upstream-${args.loggerId}`,
          data: {
            broken: [],
            message: errMsg,
          },
        });
        // ALSO emit a text part so the assistant bubble isn't empty
        // (and so persistence has something readable to save).
        const textId = "deep-error";
        writer.write({ type: "text-start", id: textId });
        writer.write({ type: "text-delta", id: textId, delta: `_${errMsg}_` });
        writer.write({ type: "text-end", id: textId });
        await persistAssistantText(`_${errMsg}_`);
        return;
      }

      const reader = upstream.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      let finalText = "";
      let finalStartedId: string | null = null;
      // What we'll persist as the assistant turn. Updated on each
      // terminating branch so the saved version matches what the
      // user saw, including error fallbacks.
      let persistedAssistantText = "";
      // Set true only when the upstream reaches the happy `final`
      // event. The Python service's `_real_run` writes a RICH
      // assistant message into conversation_messages on this branch
      // (with all tool-call blocks reconstructed from agent_events).
      // We must skip the client-side text-only PUT in that case or
      // it would clobber the server's row. Error / empty paths still
      // PUT (server didn't write, we want SOMETHING in the table).
      let serverPersistedRichAssistant = false;

      // Roll up every per-stage cost event into a single turn total —
      // the deep service emits one `cost` SSE per sub-agent invocation
      // with `{ usd, input_tokens, output_tokens, model }`. We sum,
      // then emit ONE `data-turn-cost` part at the bottom (matching
      // what regular chat already does so the UI renderer is shared).
      let totalInputTokens = 0;
      let totalOutputTokens = 0;
      let totalUsd = 0;

      const accumulateCost = (payload: unknown) => {
        if (!payload || typeof payload !== "object") return;
        const p = payload as Record<string, unknown>;
        if (typeof p.input_tokens === "number") totalInputTokens += p.input_tokens;
        if (typeof p.output_tokens === "number") totalOutputTokens += p.output_tokens;
        if (typeof p.usd === "number") totalUsd += p.usd;
      };

      const emitTurnCost = () => {
        if (totalInputTokens === 0 && totalOutputTokens === 0 && totalUsd === 0) return;
        writer.write({
          type: "data-turn-cost",
          id: `tc-deep-${args.loggerId}`,
          data: {
            model: args.model,
            input_tokens: totalInputTokens,
            output_tokens: totalOutputTokens,
            usd: totalUsd,
          },
        });
      };

      const emitStage = (kind: string, payload: unknown) => {
        writer.write({
          type: "data-deep-stage",
          id: `deep-${kind}-${Math.random().toString(36).slice(2, 10)}`,
          data: { kind, payload },
        });
      };

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += value;
          let sep: number;
          while ((sep = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            let kind = "message";
            let data: Record<string, unknown> = {};
            for (const line of frame.split("\n")) {
              if (line.startsWith("event:")) kind = line.slice(6).trim();
              else if (line.startsWith("data:")) {
                try {
                  data = JSON.parse(line.slice(5).trim()) as Record<string, unknown>;
                } catch {
                  /* keep empty */
                }
              }
            }
            if (kind === "final") {
              // Stream the final answer as a proper text part so the
              // chat bubble renders through MarkdownText (which uses
              // CitableMarkdown → ref/att/art chips work).
              finalText =
                typeof (data.payload as { text?: unknown })?.text === "string"
                  ? ((data.payload as { text: string }).text as string)
                  : finalText;
              finalStartedId = `deep-final-${args.loggerId}`;
              writer.write({ type: "text-start", id: finalStartedId });
              writer.write({ type: "text-delta", id: finalStartedId, delta: finalText });
              writer.write({ type: "text-end", id: finalStartedId });
              persistedAssistantText = finalText;
            } else if (kind === "persist_ok") {
              // Server's _persist_rich_assistant_message just COMMITTED
              // a rich assistant message into conversation_messages.
              // Setting this flag here (NOT on `final`) is the gate
              // that ensures we skip the client-side text-only PUT
              // ONLY when the server's row actually exists. If the
              // persist failed silently the server skips this frame
              // and our fallback fires below.
              serverPersistedRichAssistant = true;
            } else if (kind === "error") {
              finalStartedId = `deep-error-${args.loggerId}`;
              const reason = String((data.payload as { message?: string })?.message ?? "unknown");
              const msg = `_Deep-mode error: ${reason}_`;
              writer.write({ type: "text-start", id: finalStartedId });
              writer.write({ type: "text-delta", id: finalStartedId, delta: msg });
              writer.write({ type: "text-end", id: finalStartedId });
              persistedAssistantText = msg;
            } else if (kind === "session") {
              /* session id frame — don't surface in the UI */
            } else {
              if (kind === "cost") {
                // Per-stage cost lands inside the AssistantWork
                // disclosure (still emitted as a deep-stage) AND rolls
                // up into the bottom-of-bubble total so the user sees
                // the full turn at a glance.
                accumulateCost(data.payload ?? data);
              }
              emitStage(kind, data.payload ?? data);
            }
          }
        }
      } catch (e) {
        const err = e instanceof Error ? e.message : String(e);
        if (!finalStartedId) {
          const id = `deep-streamerr-${args.loggerId}`;
          const msg = `_Deep-mode stream error: ${err}_`;
          writer.write({ type: "text-start", id });
          writer.write({ type: "text-delta", id, delta: msg });
          writer.write({ type: "text-end", id });
          persistedAssistantText = msg;
        }
      } finally {
        try {
          reader.releaseLock();
        } catch {
          /* ignore */
        }
        // If we never produced ANY text (e.g. upstream closed mid-frame
        // before sending `final` or `error`) still persist a marker so
        // the conversation reload doesn't dead-end on a silent bubble.
        if (!persistedAssistantText) {
          persistedAssistantText =
            "_Deep-mode produced no answer — the upstream stream closed without a final frame._";
          const id = `deep-empty-${args.loggerId}`;
          writer.write({ type: "text-start", id });
          writer.write({ type: "text-delta", id, delta: persistedAssistantText });
          writer.write({ type: "text-end", id });
        }
        // Emit the bottom-of-bubble cost LAST so it sits below the
        // assistant text, mirroring regular-chat layout.
        emitTurnCost();
        // Skip persistence on the happy path — the Python service
        // already wrote a rich assistant message with full tool-call
        // detail. Re-PUTting from here would race + clobber it with
        // text-only parts (the 2026-04-27 regression). For error/
        // empty/upstream-failed paths the server didn't write, so
        // we MUST persist here or the conversation row stays stale.
        if (!serverPersistedRichAssistant) {
          await persistAssistantText(persistedAssistantText);
        }
      }
    },
  });


export async function POST(req: NextRequest) {
  const body = (await req.json()) as {
    messages?: UIMessage[];
    model?: string;
    thinkingLevel?: string;
    battle?: boolean;
    deep?: boolean;
    deep_backend?: string;
    conversation_id?: string;
  };
  const messages = body.messages;

  if (!Array.isArray(messages) || messages.length === 0) {
    return new Response(JSON.stringify({ error: "messages required" }), { status: 400 });
  }

  // Per-request overrides from the UI picker; fall back to defaults.
  const deepBackend: "adk" | "claude_code" | "claude_native" =
    body.deep_backend === "claude_code"
      ? "claude_code"
      : body.deep_backend === "claude_native"
        ? "claude_native"
        : "adk";
  // For deep mode with a Claude backend (code or native) we accept the
  // picker's model alias verbatim ("sonnet" / "opus" / etc.) — the
  // Python runtime resolves it. For everything else (chat, battle,
  // deep+adk) we restrict to the Gemini whitelist.
  const acceptModelVerbatim =
    body.deep === true && (deepBackend === "claude_code" || deepBackend === "claude_native");
  const model = acceptModelVerbatim
    ? (typeof body.model === "string" && body.model ? body.model : AGENT_MODEL)
    : isValidModel(body.model)
      ? body.model
      : AGENT_MODEL;
  const thinkingLevel: ThinkingLevel = isValidThinking(body.thinkingLevel)
    ? body.thinkingLevel
    : DEFAULT_THINKING;

  const logger = new ChatLogger();
  const question = lastUserText(messages);
  const conversationId = body.conversation_id || null;

  // Persist the user-side messages IMMEDIATELY, before any LLM work
  // starts. Without this, navigating away mid-stream meant the user's
  // typed message was never saved — saveConversation only ran inside
  // the post-completion branches, and a client disconnect aborts the
  // stream before they fire.
  //
  // Fire-and-forget on purpose — we don't want a save hiccup to delay
  // first-token latency. The post-completion save will overwrite this
  // row with the full transcript on success.
  if (conversationId) {
    const initialPersist: Array<{ role: string; parts: unknown[] }> = messages.map((m) => ({
      role: m.role,
      parts: m.parts as unknown[],
    }));
    void saveConversation(conversationId, initialPersist, deriveTitle(messages));
  }

  // Deep-mode fork: when the picker toggle is on, forward to the
  // Python multi-agent service and translate its SSE event stream
  // into a UIMessageStream the chat runtime already knows how to
  // render. Same conversation bubble shape, same chip support via
  // CitableMarkdown.
  if (body.deep) {
    console.log(`[chat ${logger.id}] deep-mode q=${JSON.stringify(question.slice(0, 80))}`);
    const stream = createDeepModeStream({
      question,
      conversationId,
      loggerId: logger.id,
      loggerPath: logger.path,
      messages,
      model,
      deepBackend,
    });
    return createUIMessageStreamResponse({ stream });
  }
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
  const tools = buildAllTools(google);
  const sqlSchema = await getSqlSchemaMarkdown();
  const system = buildSystemPrompt(tools, sqlSchema);
  const initialMessages = sanitizeConversation(await convertToModelMessages(messages));

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
      // Convergence detector: if the same broken refs come back twice in a
      // row, the model is stuck — stop hammering it.
      let lastBrokenSig: string | null = null;
      // Cumulative tool-call budget across all validation retries. Each retry
      // could otherwise burn another 15 tool calls; cap the whole turn at 25.
      const TURN_TOOL_BUDGET = 25;
      let toolCallsSoFar = 0;
      // Token totals across every validation retry for this turn — the UI
      // sees ONE `data-turn-cost` part summing the whole turn so the user
      // doesn't have to add up three separate "cost of retry #n" blurbs.
      let totalInputTokens = 0;
      let totalOutputTokens = 0;
      const emitTurnCost = () => {
        const usd = estimateCostUsd(model, totalInputTokens, totalOutputTokens);
        writer.write({
          type: "data-turn-cost",
          id: `tc-${logger.id}`,
          data: {
            model,
            input_tokens: totalInputTokens,
            output_tokens: totalOutputTokens,
            usd,
          },
        });
      };

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

        const [finalText, steps, reasoning, usage] = await Promise.all([
          result.text,
          result.steps,
          result.reasoningText,
          result.totalUsage,
        ]);

        // Accumulate across retries — one row per turn, not per attempt.
        totalInputTokens += usage.inputTokens ?? 0;
        totalOutputTokens += usage.outputTokens ?? 0;

        if (reasoning) {
          await logger.log("reasoning", { attempt: attempt + 1, text: reasoning });
        }

        for (const step of steps) {
          for (const tc of step.toolCalls ?? []) {
            toolCallsSoFar += 1;
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

        // Empty-text detection. Gemini occasionally returns no text after
        // tool-use — the UI then shows a blank assistant bubble and the
        // user assumes the server died. Surface the finish reason so they
        // know it was a model output, not an infrastructure failure.
        if (!finalText || finalText.trim().length === 0) {
          const finishReason =
            steps.length > 0 ? (steps[steps.length - 1].finishReason ?? "unknown") : "no-steps";
          const warningsText = steps
            .flatMap((s) => s.warnings ?? [])
            .map((w) => (w as { message?: string }).message ?? String(w))
            .join("; ");
          console.error(
            `[chat ${logger.id}] EMPTY TEXT attempt=${attempt + 1} finish=${finishReason} warnings=${warningsText}`,
          );
          await logger.log("empty_text", {
            attempt: attempt + 1,
            finish_reason: finishReason,
            warnings: warningsText,
          });
          const humanMsg =
            `_The model returned no text (finish reason: **${finishReason}**)._ ` +
            `This usually means a silent provider truncation or a tool-call loop that never emitted a final answer. ` +
            (warningsText ? `Provider warnings: ${warningsText}. ` : "") +
            `Try rephrasing, reducing context size (fewer attachments inlined at once), or switching models.`;
          const id = `empty-notice-${attempt}`;
          writer.write({ type: "text-start", id });
          writer.write({ type: "text-delta", id, delta: humanMsg });
          writer.write({ type: "text-end", id });
          await logger.log("done", { attempts: attempt + 1, ms: logger.elapsedMs() });
          if (conversationId) {
            const persisted: Array<{ role: string; parts: unknown[] }> = messages.map((m) => ({
              role: m.role,
              parts: m.parts as unknown[],
            }));
            persisted.push({
              role: "assistant",
              parts: [{ type: "text", text: humanMsg }],
            });
            await saveConversation(conversationId, persisted, deriveTitle(messages));
          }
          emitTurnCost();
          return;
        }
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

        // Convergence check: if the model produced the EXACT same broken
        // refs as last attempt, more retries won't help — bail with the
        // warning and stop burning tokens.
        const brokenSig = broken.length > 0 ? broken.slice().sort().join(",") : null;
        const stuck = brokenSig !== null && brokenSig === lastBrokenSig;
        const overBudget = toolCallsSoFar > TURN_TOOL_BUDGET;
        lastBrokenSig = brokenSig;

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
          emitTurnCost();
          return;
        }
        if (attempt === MAX_VALIDATION_RETRIES || stuck || overBudget) {
          const reason = stuck
            ? "model produced the same broken refs again — stopped retrying"
            : overBudget
              ? `tool-call budget exhausted (${toolCallsSoFar}/${TURN_TOOL_BUDGET})`
              : `${MAX_VALIDATION_RETRIES + 1} attempts exhausted`;
          // For non-final attempts, the text was suppressed — replay it now
          // so the user at least sees the partially-cited answer.
          if (!isFinalAttempt) {
            const id = `final-text-${attempt}`;
            writer.write({ type: "text-start", id });
            writer.write({ type: "text-delta", id, delta: finalText });
            writer.write({ type: "text-end", id });
          }
          writer.write({
            type: "data-citation-warning",
            id: `cw-${attempt}`,
            data: {
              broken,
              message: `Could not validate ${broken.length} citation(s): ${reason}.`,
            },
          });
          await logger.log("done", {
            attempts: attempt + 1,
            ms: logger.elapsedMs(),
            unresolved_broken: broken,
            stopped_for: stuck ? "convergence" : overBudget ? "budget" : "max_attempts",
          });
          emitTurnCost();
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
      return friendlyModelError(msg);
    },
  });

  return createUIMessageStreamResponse({ stream });
}
