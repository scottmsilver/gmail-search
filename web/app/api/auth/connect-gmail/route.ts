import { publicRoute } from "@/lib/publicBoundary";
// Proxies /api/auth/connect-gmail. Same shape as /api/auth/login —
// returns 307 + state cookie so the browser navigates to the broker
// with Gmail+Drive scope. The broker round-trip lands on the same
// /api/auth/callback we use for sign-in, which is idempotent and
// just refreshes the session.

import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

async function handleGET(req: NextRequest) {
  const search = req.nextUrl.search;
  const cookie = req.headers.get("cookie") ?? "";
  const fwdHost = req.headers.get("host") ?? "";
  const fwdProto = process.env.GMS_PUBLIC_ORIGIN !== undefined ? "https" : req.nextUrl.protocol.replace(":", "");
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

export const GET = publicRoute(handleGET);

async function handlePOST(req: NextRequest) {
  const {rejectUnsafeOrigin, backendOriginHeaders} = await import('@/lib/originSecurity');
  const denied = rejectUnsafeOrigin(req);
  if (denied) return denied;
  if (process.env.GMS_FULL_WORKER_ROUTES !== '1') return new Response('Not found', {status:404});
  if (new URL(req.url).search || (await req.text()).trim()) {
    return new Response('Unexpected consent parameters', {status:400});
  }
  const upstream = await fetch(`${pythonApiUrl()}/api/auth/connect-gmail`, {
    method:'POST', redirect:'manual', cache:'no-store',
    headers:{...backendOriginHeaders(), cookie:req.headers.get('cookie') ?? '', 'Content-Type':'application/json'},
    body:'{}',
  });
  const headers = new Headers();
  for (const name of ['location','content-type']) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name,value);
  }
  for (const value of upstream.headers.getSetCookie()) headers.append('set-cookie',value);
  return new NextResponse(upstream.body,{status:upstream.status,headers});
}
export const POST = publicRoute(handlePOST);
