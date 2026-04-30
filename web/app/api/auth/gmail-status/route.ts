// Proxies /api/auth/gmail-status. Forwards the session cookie so the
// FastAPI side can identify which user we're asking about. Returns
// {multi_tenant, connected, scope_problem?}.

import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function GET(req: NextRequest) {
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/auth/gmail-status`, {
    method: "GET",
    cache: "no-store",
    headers: cookie ? { cookie } : undefined,
  });
  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
