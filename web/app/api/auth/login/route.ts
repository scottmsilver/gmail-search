// Proxies /api/auth/login (Next.js side) → FastAPI's /api/auth/login.
//
// Critical detail: FastAPI returns 307 + a Set-Cookie (the OAuth state
// cookie). We must NOT follow the redirect (default fetch behaviour
// would do that), and we must relay both the Location header and the
// Set-Cookie header verbatim so the browser:
//   1. Sets the state cookie on this origin (localhost:3000 in dev,
//      gms.i.oursilverfamily.com in prod).
//   2. Navigates to the broker.
//
// Without this proxy, browser navigation to /api/auth/login on the
// Next.js port (3000) would 404 in dev. In production behind Caddy
// the routing already fans /api/* to FastAPI, but the proxy is
// harmless there too.

import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function GET(req: NextRequest) {
  const search = req.nextUrl.search; // includes leading '?' or empty
  // Tell FastAPI which host the browser actually sees, so the inner
  // /api/auth/callback URL it hands to the broker points back at us
  // (the Next.js origin), not at FastAPI's upstream port. Without this
  // the broker would round-trip the user to 127.0.0.1:8090 and the
  // state + session cookies would land on the wrong origin.
  const fwdHost = req.headers.get("host") ?? "";
  const fwdProto = req.nextUrl.protocol.replace(":", "");
  const upstream = await fetch(`${pythonApiUrl()}/api/auth/login${search}`, {
    method: "GET",
    redirect: "manual",
    cache: "no-store",
    headers: {
      "x-forwarded-host": fwdHost,
      "x-forwarded-proto": fwdProto,
    },
  });

  const headers = new Headers();
  const location = upstream.headers.get("location");
  if (location) headers.set("location", location);
  for (const c of upstream.headers.getSetCookie()) headers.append("set-cookie", c);

  // 307 from upstream; preserve verbatim. (RedirectResponse on FastAPI
  // returns 307; some Caddy setups normalise to 302 — both are valid.)
  return new NextResponse(null, { status: upstream.status, headers });
}
