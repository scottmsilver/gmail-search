import { publicRoute } from "@/lib/publicBoundary";
import { backendOriginHeaders, rejectUnsafeOrigin } from "@/lib/originSecurity";
// Proxies /api/auth/logout. Forwards the inbound cookie so FastAPI
// can identify the session, relays the Set-Cookie that clears it.

import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

async function handlePOST(req: NextRequest) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/auth/logout`, {
    method: "POST",
    cache: "no-store",
    headers: { ...backendOriginHeaders(), ...(cookie ? { cookie } : {}) },
  });
  const body = await upstream.text();
  const headers = new Headers({
    "content-type": upstream.headers.get("content-type") ?? "application/json",
  });
  for (const c of upstream.headers.getSetCookie()) headers.append("set-cookie", c);
  return new NextResponse(body, { status: upstream.status, headers });
}

export const POST = publicRoute(handlePOST);
