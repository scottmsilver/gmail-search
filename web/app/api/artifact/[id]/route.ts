import { publicRoute } from "@/lib/publicBoundary";
import { type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

// Proxy for /api/artifact/<id>. The Writer cites analyst-produced
// plots / CSVs as [art:<id>]; the UI resolves that chip by hitting
// this route, which forwards to the Python endpoint. Bytes flow
// straight through with the original content-type preserved so
// images render inline and CSVs download.
import { downloadHeaders } from "@/lib/requestSecurity";

export const runtime = "nodejs";

async function handleGET(req: NextRequest, ctx: { params: Promise<{ id: string }> }): Promise<Response> {
  const { id } = await ctx.params;
  if (!/^(?:[a-f0-9]{32}|[0-9]+)$/.test(id)) return new Response("Not found", {status: 404});
  const cookie = req.headers.get("cookie") ?? "";
  let url = `${pythonApiUrl()}/api/artifact/${encodeURIComponent(id)}`;
  if (/^[a-f0-9]{32}$/.test(id)) {
    if (process.env.GMS_FULL_WORKER_ROUTES !== "1") return new Response("Not found", {status: 404});
    const values = new URL(req.url).searchParams;
    const conversation = values.get("conversation_id");
    if (!conversation || !/^[a-zA-Z0-9_-]{1,256}$/.test(conversation)
        || values.getAll("conversation_id").length !== 1
        || [...values.keys()].some(key => key !== "conversation_id")) {
      return new Response("Conversation required", {status: 400});
    }
    url = `${pythonApiUrl()}/api/agent-artifacts/${id}?conversation_id=${encodeURIComponent(conversation)}`;
  }
  const upstream = await fetch(url, {
    cache: "no-store",
    headers: cookie ? { cookie } : undefined,
  });
  if (!upstream.ok) {
    return new Response(`artifact ${id} not found`, { status: upstream.status, headers: downloadHeaders(upstream.headers) });
  }
  const headers = downloadHeaders(upstream.headers);
  return new Response(upstream.body, { status: 200, headers });
}

export const GET = publicRoute(handleGET);
