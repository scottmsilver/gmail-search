import { workerChatHistory } from "@/lib/workerChatHistory";
import { isPublicMode, publicRoute } from "@/lib/publicBoundary";
import { createUIMessageStream, createUIMessageStreamResponse, type UIMessage } from "ai";
import type { NextRequest } from "next/server";
import { availableModelsFor, pythonApiUrl, piModelForRequest, type DeepBackend } from "@/lib/config";
import { pickTwoRandomVariants } from "@/lib/battleVariants";
import { battleQuestion, emptyBattleSide, runDeepBattleSide } from "@/lib/deepBattle";
import { ChatLogger } from "@/lib/chatLog";

import { authenticatedCaller, unauthenticated, privateHeaders } from "@/lib/requestSecurity";

import { rejectUnsafeOrigin, backendOriginHeaders } from "@/lib/originSecurity";
import { servedChoice } from "@/lib/deepModels";

export const runtime = "nodejs";
export const maxDuration = 1800;

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
  cookie: string,
): Promise<boolean> => {
  try {
    const headers: Record<string, string> = { ...backendOriginHeaders(), "Content-Type": "application/json" };
    if (cookie) headers["cookie"] = cookie;
    const res = await fetch(
      `${pythonApiUrl()}/api/conversations/${encodeURIComponent(conversationId)}`,
      {
        method: "PUT",
        headers,
        body: JSON.stringify({ title, messages }),
      },
    );
    if (!res.ok) {
      console.error(`[chat] save conversation ${conversationId} failed: ${res.status}`);
      return false;
    }
    return true;
  } catch (err) {
    console.error(`[chat] save conversation ${conversationId} threw:`, err);
    return false;
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

type DeepStreamArgs = {
  question: string;
  conversationId: string | null;
  loggerId: string;
  loggerPath: string;
  ownerId: string;
  // Full inbound messages so we can persist the user turn alongside
  // the assistant's deep-mode reply on conversation save. Without
  // this the user reloads the conversation and sees only the half
  // they typed (or worse, nothing).
  messages: UIMessage[];
  // Picker-selected model — applied to every deep sub-agent
  // (Planner / Retriever / Analyst / Writer / Critic). When omitted,
  // Python falls back to per-stage env vars / built-in defaults.
  model?: string;
  // Runtime selected for this deep-analysis request.
  deepBackend?: DeepBackend;
  // Inbound session cookie — forwarded to FastAPI so require_user_id
  // can resolve which user this turn belongs to.
  cookie: string;
  signal: AbortSignal;
};

const createDeepModeStream = (args: DeepStreamArgs) =>
  createUIMessageStream({
    execute: async ({ writer }) => {
      const logger = new ChatLogger(args.loggerId, args.ownerId);
      await logger.log("request", {
        conversation_id: args.conversationId, model: args.model,
        backend: args.deepBackend, question: args.question,
      });
      try {
      if (!isPublicMode()) writer.write({
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
        await saveConversation(args.conversationId, persisted, deriveTitle(args.messages), args.cookie);
      };

      const upstreamHeaders: Record<string, string> = { ...backendOriginHeaders(), "Content-Type": "application/json" };
      if (args.cookie) upstreamHeaders["cookie"] = args.cookie;
      const upstream = await fetch(`${pythonApiUrl()}/api/agent/analyze`, {
        method: "POST",
        signal: args.signal,
        headers: upstreamHeaders,
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
        await logger.log("error", { message: errMsg });
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

      // The full worker controller owns durable transcripts even if the
      // connection drops before its first session or final persist_ok frame.
      let serverOwnsTranscript = upstream.headers.get("X-GMS-Transcript-Owner") === "server";
      const reader = upstream.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      let finalText = "";
      let workerRunId: string | null = null;
      let finalStartedId: string | null = null;
      // Answer text streamed while the model writes it (`answer_delta`); the
      // `final` frame then only closes it, unless the texts differ.
      const streamed = {
        id: `deep-stream-${args.loggerId}`, text: "", open: false,
        append(delta: string) {
          if (!this.open) { writer.write({ type: "text-start", id: this.id }); this.open = true; }
          writer.write({ type: "text-delta", id: this.id, delta });
          this.text += delta;
        },
        end() { if (this.open) { writer.write({ type: "text-end", id: this.id }); this.open = false; } },
        matches(final: string) { return this.text.trim() !== "" && this.text.trim() === final.trim(); },
      };
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
            await logger.log(kind === "error" ? "error" : "event", { event: kind, ...data });
            if (kind === "final") {
              // Stream the final answer as a proper text part so the
              // chat bubble renders through MarkdownText (which uses
              // CitableMarkdown → ref/att/art chips work).
              finalText =
                typeof (data.payload as { text?: unknown })?.text === "string"
                  ? ((data.payload as { text: string }).text as string)
                  : finalText;
              streamed.end();
              if (streamed.matches(finalText)) {
                finalStartedId = streamed.id;
              } else {
                finalStartedId = `deep-final-${args.loggerId}`;
                writer.write({ type: "text-start", id: finalStartedId });
                writer.write({ type: "text-delta", id: finalStartedId, delta: finalText });
                writer.write({ type: "text-end", id: finalStartedId });
              }
              persistedAssistantText = finalText;
            } else if (kind === "answer_delta") {
              const delta = (data.payload as { text?: unknown })?.text;
              if (typeof delta === "string" && delta) streamed.append(delta);
            } else if (kind === "persist_ok") {
              // Server's _persist_rich_assistant_message just COMMITTED
              // a rich assistant message into conversation_messages.
              // Setting this flag here (NOT on `final`) is the gate
              // that ensures we skip the client-side text-only PUT
              // ONLY when the server's row actually exists. If the
              // persist failed silently the server skips this frame
              // and our fallback fires below.
              serverPersistedRichAssistant = true;
              if (workerRunId) writer.write({type: "data-agent-run-finished", transient: true,
                data: {runId: workerRunId}});
            } else if (kind === "error") {
              streamed.end();
              finalStartedId = `deep-error-${args.loggerId}`;
              const reason = String((data.payload as { message?: string })?.message ?? "unknown");
              const state = (data.payload as {state?: unknown})?.state;
              if (workerRunId && (state === "cancelled" || state === "failed")) {
                writer.write({type: "data-agent-run-finished", transient: true, data: {runId: workerRunId}});
              }
              const msg = `_Deep-mode error: ${reason}_`;
              writer.write({ type: "text-start", id: finalStartedId });
              writer.write({ type: "text-delta", id: finalStartedId, delta: msg });
              writer.write({ type: "text-end", id: finalStartedId });
              persistedAssistantText = msg;
            } else if (kind === "session") {
              if (data.supports_cancel === true && typeof data.session_id === "string"
                  && typeof data.conversation_id === "string") {
                serverOwnsTranscript = true;
                workerRunId = data.session_id;
                writer.write({type: "data-agent-run", transient: true,
                  data: {runId: data.session_id, conversationId: data.conversation_id}});
                // The isolated runtime's first event waits on the VM booting
                // (~4s). Show the run as started now, not after that wait.
                emitStage("status", {type: "status", state: "starting"});
              }
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
        await logger.log("error", { message: err });
        streamed.end();
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
          await logger.log("empty_text", { message: "Upstream closed without a final answer" });
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
        if (!serverOwnsTranscript && !serverPersistedRichAssistant) {
          await persistAssistantText(persistedAssistantText);
        }
      }
      } catch (error) {
        await logger.log("error", { message: error instanceof Error ? error.message : String(error) });
        throw error;
      } finally {
        await logger.log("done", { elapsed_ms: logger.elapsedMs() });
      }
    },
  });


async function handlePOST(req: NextRequest) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const caller = await authenticatedCaller(req);
  if (!caller) return unauthenticated();
  const ownerId = caller.id;
  // Capability, not deployment. The public origin restricts by default; a caller
  // the server reports as capable keeps model choice, battles and backend
  // selection. Decided here from the authenticated probe, never from the body.
  const restricted = isPublicMode() && !caller.fullRuntime;
  const body = await req.json() as {
    messages?: UIMessage[]; model?: string; battle?: boolean;
    deep_backend?: DeepBackend; conversation_id?: string;
  };
  const served = caller.deepModels;
  if ((restricted || served) && body.battle) return Response.json({error: "Battle mode is unavailable on the public app"}, {status: 400});
  const messages = body.messages;
  if (!Array.isArray(messages) || !messages.length) {
    return Response.json({error: "messages required"}, {status: 400});
  }
  if (isPublicMode() && (messages.length > 100 || messages.some(m => !m || !["user", "assistant"].includes(m.role) || !Array.isArray(m.parts) || m.parts.some(p => !p || typeof p !== "object" || (p.type === "text" && typeof p.text !== "string"))))) {
    return Response.json({error: "Invalid chat messages"}, {status: 400});
  }
  const question = lastUserText(messages);
  if (!question.trim()) return Response.json({error: "question required"}, {status: 400});
  if (isPublicMode() && (question.length > 16000 || messages.length > 100)) return Response.json({error: "Chat request too large"}, {status: 413});
  // The server-reported served table wins: its choice is what actually runs.
  const servedPick = served && servedChoice(served, body.deep_backend, body.model);
  const backend: DeepBackend = servedPick ? servedPick.backend : restricted ? "pi" : ["claude_code", "pi"].includes(body.deep_backend ?? "")
    ? body.deep_backend! : "pi";
  const models = availableModelsFor(backend);
  const model = servedPick ? servedPick.model : restricted ? undefined : backend === "pi" ? piModelForRequest(body.model)
    : models.includes(body.model ?? "") ? body.model : models[0];
  const logger = new ChatLogger(undefined, ownerId);
  const conversationId = body.conversation_id || null;
  const cookie = req.headers.get("cookie") ?? "";
  if (conversationId) {
    const saved = await saveConversation(conversationId, (process.env.GMS_FULL_WORKER_ROUTES === "1" ? workerChatHistory(messages) : messages).map(m => ({role: m.role, parts: m.parts})), deriveTitle(messages), cookie);
    if (!saved) return Response.json({error: "Conversation could not be saved. Please retry."}, {status: 502});
  }
  if (!body.battle) return createUIMessageStreamResponse({headers: privateHeaders, stream: createDeepModeStream({
    question, conversationId, loggerId: logger.id, loggerPath: logger.path, ownerId, messages, model, deepBackend: backend, cookie, signal: req.signal,
  })});

  const [variantA, variantB] = pickTwoRandomVariants();
  const id = `battle-${logger.id}`;
  const stream = createUIMessageStream({execute: async ({writer}) => {
    let a = emptyBattleSide();
    let b = emptyBattleSide();
    const data = () => ({
      request_id: logger.id, question, variant_a: variantA, variant_b: variantB,
      answer_a: a.error ? `${a.answer}\n\n⚠ ${a.error}`.trim() : a.answer,
      answer_b: b.error ? `${b.answer}\n\n⚠ ${b.error}`.trim() : b.answer,
      tools_a: a.running ? [] : a.tools, tools_b: b.running ? [] : b.tools, running_a: a.running, running_b: b.running,
      steps_a: a.steps, steps_b: b.steps, usd_a: a.usd, usd_b: b.usd,
      session_a: a.sessionId, session_b: b.sessionId,
    });
    const emit = () => writer.write({type: "data-battle", id, data: data()});
    emit();
    const common = {url: `${pythonApiUrl()}/api/agent/analyze`, cookie, question: battleQuestion(messages)};
    await Promise.all([
      runDeepBattleSide({...common, variant: variantA, onUpdate: state => {a = state; emit();}}),
      runDeepBattleSide({...common, variant: variantB, onUpdate: state => {b = state; emit();}}),
    ]);
    const final = data();
    try { await logger.log("battle_done", final); } catch (error) {
      console.error("Battle logging failed:", error);
    }
    if (conversationId) {
      const saved = await saveConversation(conversationId, [
        ...messages.map(m => ({role: m.role, parts: m.parts as unknown[]})),
        {role: "assistant", parts: [{type: "data-battle", id, data: final}]},
      ], deriveTitle(messages), cookie);
      if (!saved) writer.write({type: "data-citation-warning", id: `save-${id}`, data: {
        broken: [], message: "The battle finished, but saving failed. Copy the answers before leaving this page.",
      }});
    }
  }});
  return createUIMessageStreamResponse({headers: privateHeaders, stream});
}

export const POST = publicRoute(handlePOST);
