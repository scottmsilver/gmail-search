// Proxies /api/auth/connect-gmail. Same shape as /api/auth/login —
// returns 307 + state cookie so the browser navigates to the broker
// with Gmail+Drive scope. The broker round-trip lands on the same
// /api/auth/callback we use for sign-in, which is idempotent and
// just refreshes the session.

import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function GET(req: NextRequest) {
  const search = req.nextUrl.search;
  const cookie = req.headers.get("cookie") ?? "";
  const fwdHost = req.headers.get("host") ?? "";
  const fwdProto = req.nextUrl.protocol.replace(":", "");
  const fwdHeaders: Record<string, string> = {
    "x-forwarded-host": fwdHost,
    "x-forwarded-proto": fwdProto,
  };
  if (cookie) fwdHeaders.cookie = cookie;
  const upstream = await fetch(`${pythonApiUrl()}/api/auth/connect-gmail${search}`, {
    method: "GET",
    redirect: "manual",
    cache: "no-store",
    headers: fwdHeaders,
  });

  const respHeaders = new Headers();
  const location = upstream.headers.get("location");
  if (location) respHeaders.set("location", location);
  for (const c of upstream.headers.getSetCookie()) respHeaders.append("set-cookie", c);

  // Forward body in all cases — 4xx/5xx need the detail field, 3xx
  // can carry a small "Redirecting..." HTML payload.
  const body = await upstream.text();
  const ct = upstream.headers.get("content-type");
  if (ct) respHeaders.set("content-type", ct);
  return new NextResponse(body, { status: upstream.status, headers: respHeaders });
}
