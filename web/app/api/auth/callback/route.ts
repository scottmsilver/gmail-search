// Proxies /api/auth/callback. The browser arrives here from the
// silver-oauth broker carrying ?silver_oauth=…&state=…&return=…
// PLUS the gms_oauth_state cookie we set in /api/auth/login.
//
// We must:
//   * Forward the inbound Cookie header so FastAPI sees the state
//     cookie (otherwise the state binding check 400s).
//   * NOT follow the upstream 307 redirect (we'd lose the Location
//     header and the Set-Cookie that mints the session).
//   * Relay every Set-Cookie verbatim — there are TWO: one clears
//     the state cookie, one sets the session cookie.

import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function GET(req: NextRequest) {
  const search = req.nextUrl.search;
  const cookie = req.headers.get("cookie") ?? "";
  const fwdHost = req.headers.get("host") ?? "";
  const fwdProto = req.nextUrl.protocol.replace(":", "");
  // Same forwarding game as /api/auth/login — FastAPI uses these to
  // decide whether to flag the session cookie Secure (HTTPS only).
  const fwdHeaders: Record<string, string> = {
    "x-forwarded-host": fwdHost,
    "x-forwarded-proto": fwdProto,
  };
  if (cookie) fwdHeaders.cookie = cookie;
  const upstream = await fetch(`${pythonApiUrl()}/api/auth/callback${search}`, {
    method: "GET",
    redirect: "manual",
    cache: "no-store",
    headers: fwdHeaders,
  });

  const respHeaders = new Headers();
  const location = upstream.headers.get("location");
  if (location) respHeaders.set("location", location);
  for (const c of upstream.headers.getSetCookie()) respHeaders.append("set-cookie", c);

  // Forward the body in all cases — RedirectResponse from FastAPI
  // sends a small "Redirecting..." HTML payload some clients/log
  // analyzers expect on 3xx, and error paths (4xx/5xx) need the
  // detail so the user sees a useful error page.
  const body = await upstream.text();
  const upstreamCT = upstream.headers.get("content-type");
  if (upstreamCT) respHeaders.set("content-type", upstreamCT);
  return new NextResponse(body, { status: upstream.status, headers: respHeaders });
}
