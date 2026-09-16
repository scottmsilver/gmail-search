import { publicRoute } from "@/lib/publicBoundary";
import { backendOriginHeaders, rejectUnsafeOrigin } from "@/lib/originSecurity";
import { type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

// SSE proxy to the Python deep-analysis endpoint. Kept thin on
// purpose — body is JSON in, SSE stream out, with back-pressure
// handled by pipeThrough() rather than any buffering in the Node
// middle. The Python side handles session creation, orchestration,
// event persistence; we just forward bytes.
export const runtime = "nodejs";

async function handlePOST(req: NextRequest): Promise<Response> {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const bodyText = await req.text();
  const cookie = req.headers.get("cookie") ?? "";
  const headers: Record<string, string> = { ...backendOriginHeaders(), "Content-Type": "application/json" };
  if (cookie) headers["cookie"] = cookie;
  const upstream = await fetch(`${pythonApiUrl()}/api/agent/analyze`, {
    method: "POST",
    signal: req.signal,
    headers,
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
      "Cache-Control": "private, no-store",
      Connection: "keep-alive",
    },
  });
}

// Replay stream for reconnection — mirrors the Python /events
// endpoint. UI calls this with ?after=<last_seq> after a drop to
// resume from where it left off without missing events.
async function handleGET(req: NextRequest): Promise<Response> {
  const sessionId = req.nextUrl.searchParams.get("session_id");
  const after = req.nextUrl.searchParams.get("after") ?? "0";
  if (!sessionId) {
    return new Response("session_id required", { status: 400 });
  }
  const url = new URL(`${pythonApiUrl()}/api/agent/analyze/${encodeURIComponent(sessionId)}/events`);
  url.searchParams.set("after", after);
  const conversation = req.nextUrl.searchParams.get("conversation_id");
  if (conversation) url.searchParams.set("conversation_id", conversation);
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(url.toString(), {
    signal: req.signal,
    headers: { ...backendOriginHeaders(), ...(cookie ? { cookie } : {}) },
  });
  if (!upstream.body) {
    return new Response("upstream returned no body", { status: 502 });
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "private, no-store",
      Connection: "keep-alive",
    },
  });
}

export const POST = publicRoute(handlePOST);

export const GET = publicRoute(handleGET);

// Stop has its own request lifetime; aborting the answer stream must not
// abort cleanup before the controller acknowledges worker termination.
async function handleDELETE(req: NextRequest): Promise<Response> {
  const denied = rejectUnsafeOrigin(req);
  if (denied) return denied;
  const body = await req.json();
  if (!body || typeof body !== "object" || Array.isArray(body)
      || Object.keys(body).some(key => !["run_id", "conversation_id"].includes(key))
      || typeof body.run_id !== "string" || !/^[a-zA-Z0-9_-]{1,256}$/.test(body.run_id)
      || typeof body.conversation_id !== "string" || !/^[a-zA-Z0-9_-]{1,256}$/.test(body.conversation_id)) {
    return Response.json({error: "Run and conversation required"}, {status: 400});
  }
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/agent/analyze/${encodeURIComponent(body.run_id)}/cancel`, {
    method: "POST",
    headers: {...backendOriginHeaders(), "Content-Type": "application/json", ...(cookie ? {cookie} : {})},
    body: JSON.stringify({conversation_id: body.conversation_id}),
  });
  return new Response(upstream.body, {status: upstream.status, headers: {
    "Content-Type": "application/json", "Cache-Control": "private, no-store",
  }});
}
export const DELETE = publicRoute(handleDELETE);
