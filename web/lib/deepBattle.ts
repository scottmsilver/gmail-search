import { backendOriginHeaders } from "./originSecurity";
import type { UIMessage } from "ai";
import type { BattleVariant } from "./battleVariants";

export type BattleSideState = {
  answer: string;
  error?: string;
  running: boolean;
  steps: number;
  usd: number;
  sessionId?: string;
  tools: Array<{ name: string; args: unknown; output: unknown }>;
};
export const emptyBattleSide = (): BattleSideState => ({answer: "", running: true, steps: 0, usd: 0, tools: []});

// No shared conversation/session ID: both agents start fresh with identical context.
export const battleQuestion = (messages: UIMessage[]): string => {
  const text = (m: UIMessage) => m.parts.flatMap(part => {
    if (part.type === "text") return [part.text];
    if (part.type === "data-battle") {
      const data = part.data as {answer_a?: string; answer_b?: string};
      return [`Previous analysis A: ${data.answer_a ?? ""}`, `Previous analysis B: ${data.answer_b ?? ""}`];
    }
    return [];
  }).join("\n");
  const last = messages.findLastIndex(m => m.role === "user");
  if (last < 0) return "";
  const history = messages.slice(0, last).map(m => `${m.role}: ${text(m)}`).join("\n\n").slice(-24000);
  return `${history ? `Prior conversation (context only):\n${history}\n\n` : ""}Current question:\n${text(messages[last])}`;
};

export const runDeepBattleSide = async (args: {
  url: string; cookie: string; question: string; variant: BattleVariant;
  onUpdate: (state: BattleSideState) => void;
  fetchImpl?: typeof fetch;
}): Promise<BattleSideState> => {
  const state = emptyBattleSide();
  let reader: ReadableStreamDefaultReader<string> | undefined;
  const update = () => args.onUpdate({...state, tools: [...state.tools]});
  try {
    const response = await (args.fetchImpl ?? fetch)(args.url, {
      method: "POST",
      headers: {...backendOriginHeaders(), "Content-Type": "application/json", ...(args.cookie ? {cookie: args.cookie} : {})},
      body: JSON.stringify({question: args.question, backend: args.variant.backend, model: args.variant.model}),
      signal: AbortSignal.timeout(30 * 60 * 1000),
    });
    if (!response.ok || !response.body) throw new Error(`Deep analysis failed (HTTP ${response.status}): ${(await response.text()).slice(0, 500)}`);
    reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    const frame = (raw: string) => {
      let kind = "message";
      const lines: string[] = [];
      for (const line of raw.split(/\r?\n/)) {
        if (line.startsWith("event:")) kind = line.slice(6).trim();
        if (line.startsWith("data:")) lines.push(line.slice(5).trim());
      }
      if (!lines.length) return;
      const record = JSON.parse(lines.join("\n"));
      const payload = record.payload ?? record;
      if (kind === "session") state.sessionId = record.session_id;
      else if (kind === "final") state.answer = typeof payload.text === "string" ? payload.text : "";
      else if (kind === "error") state.error = String(payload.message ?? "Deep analysis failed");
      else if (kind === "cost") state.usd += typeof payload.usd === "number" ? payload.usd : 0;
      else if (kind !== "persist_ok") {
        state.steps++;
        // Retain stage details for the inspector after the battle is complete.
        if (state.tools.length < 500) state.tools.push({name: kind, args: {}, output: payload});
      }
      update();
    };
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += value;
      let boundary: RegExpExecArray | null;
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        frame(buffer.slice(0, boundary.index));
        buffer = buffer.slice(boundary.index + boundary[0].length);
      }
    }
    if (buffer.trim()) frame(buffer);
    if (!state.answer.trim() && !state.error) state.error = "Deep analysis ended without a final answer.";
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    if (reader) {
      try { await reader.cancel(); } catch { /* stream already closed */ }
      reader.releaseLock();
    }
    state.running = false;
    update();
  }
  return state;
};
