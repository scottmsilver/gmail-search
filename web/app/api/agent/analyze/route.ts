import { type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

// SSE proxy to the Python deep-analysis endpoint. Kept thin on
// purpose — body is JSON in, SSE stream out, with back-pressure
// handled by pipeThrough() rather than any buffering in the Node
// middle. The Python side handles session creation, orchestration,
// event persistence; we just forward bytes.
export const runtime = "nodejs";

export async function POST(req: NextRequest): Promise<Response> {
  const bodyText = await req.text();
  const upstream = await fetch(`${pythonApiUrl()}/api/agent/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: bodyText,
  });
  if (!upstream.body) {
    return new Response("upstream returned no body", { status: 502 });
  }
  // Pass the event-stream through as-is. Next.js streams this to
  // the browser; EventSource / fetch-reader on the client
  // consumes frames directly.
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

// Replay stream for reconnection — mirrors the Python /events
// endpoint. UI calls this with ?after=<last_seq> after a drop to
// resume from where it left off without missing events.
export async function GET(req: NextRequest): Promise<Response> {
  const sessionId = req.nextUrl.searchParams.get("session_id");
  const after = req.nextUrl.searchParams.get("after") ?? "0";
  if (!sessionId) {
    return new Response("session_id required", { status: 400 });
  }
  const url = new URL(`${pythonApiUrl()}/api/agent/analyze/${encodeURIComponent(sessionId)}/events`);
  url.searchParams.set("after", after);
  const upstream = await fetch(url.toString());
  if (!upstream.body) {
    return new Response("upstream returned no body", { status: 502 });
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
