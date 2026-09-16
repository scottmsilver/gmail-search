import { publicRoute } from "@/lib/publicBoundary";
import { type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

import { downloadHeaders } from "@/lib/requestSecurity";

export const runtime = "nodejs";
export const revalidate = 0;

// Streams the binary attachment from the Python backend so the browser
// only ever talks to one origin. Lets us serve everything behind one
// hostname (gms.i.oursilverfamily.com) without exposing the Python
// backend port directly.
async function handleGET(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/attachment/${encodeURIComponent(id)}`, {
    cache: "no-store",
    headers: cookie ? { cookie } : undefined,
  });
  const headers = downloadHeaders(upstream.headers);
  return new Response(upstream.body, { status: upstream.status, headers });
}

export const GET = publicRoute(handleGET);
