// Tiny SSE parser for the /api/agent/analyze POST-then-stream route.
// Browsers' built-in EventSource is GET-only and doesn't support a
// POST body, so we roll our own: fetch() the stream, pipe through a
// TextDecoder, split on blank lines, parse `event:` + `data:` frames.
//
// The server-side SSE frames the Python orchestrator writes are
// shaped:  event: <kind>\ndata: <json>\n\n
// Kind is one of:  session | plan | retriever (evidence) | analyst
//                   | draft | critique | revision | final | error
// The raw payload shape varies per kind — callers discriminate on
// `kind` and read `seq`, `agent`, `payload`.

export type AgentEvent = {
  kind: string;
  seq?: number;
  agent?: string;
  payload?: unknown;
  // Fallback slot for events that don't carry seq/agent/payload —
  // currently only the first `session` frame, which carries
  // `{session_id: "..."}` directly.
  raw?: Record<string, unknown>;
};

export type AgentStreamHandler = {
  onEvent?: (event: AgentEvent) => void;
  onError?: (err: unknown) => void;
  onDone?: () => void;
  signal?: AbortSignal;
};

/**
 * POST `{question}` to /api/agent/analyze and stream events back.
 * Returns a promise that resolves when the stream closes. Per-event
 * handling is via callbacks so the caller can update UI state
 * incrementally as each stage completes.
 */
export const runAgentAnalyze = async (
  body: { question: string; model?: string; conversation_id?: string },
  handlers: AgentStreamHandler = {},
): Promise<void> => {
  const resp = await fetch("/api/agent/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: handlers.signal,
  });
  if (!resp.ok || !resp.body) {
    handlers.onError?.(new Error(`agent stream failed: ${resp.status}`));
    return;
  }
  const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      // Frames are delimited by a blank line. Flush every complete
      // frame and keep the trailing partial in the buffer.
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const event = parseFrame(frame);
        if (event) handlers.onEvent?.(event);
      }
    }
    handlers.onDone?.();
  } catch (e) {
    handlers.onError?.(e);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
};

const parseFrame = (frame: string): AgentEvent | null => {
  let kind = "message";
  let dataLine = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) kind = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
  }
  if (!dataLine) return { kind };
  try {
    const parsed = JSON.parse(dataLine) as Record<string, unknown>;
    return {
      kind,
      seq: typeof parsed.seq === "number" ? parsed.seq : undefined,
      agent: typeof parsed.agent === "string" ? parsed.agent : undefined,
      payload: parsed.payload,
      raw: parsed,
    };
  } catch {
    return { kind, raw: { data: dataLine } };
  }
};
