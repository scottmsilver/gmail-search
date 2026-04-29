// Proxies /api/auth/logout. Forwards the inbound cookie so FastAPI
// can identify the session, relays the Set-Cookie that clears it.

import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function POST(req: NextRequest) {
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/auth/logout`, {
    method: "POST",
    cache: "no-store",
    headers: cookie ? { cookie } : undefined,
  });
  const body = await upstream.text();
  const headers = new Headers({
    "content-type": upstream.headers.get("content-type") ?? "application/json",
  });
  for (const c of upstream.headers.getSetCookie()) headers.append("set-cookie", c);
  return new NextResponse(body, { status: upstream.status, headers });
}
