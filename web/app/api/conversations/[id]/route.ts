import { publicRoute } from "@/lib/publicBoundary";
import { backendOriginHeaders, rejectUnsafeOrigin } from "@/lib/originSecurity";
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

const ID_RE = /^[a-zA-Z0-9_-]{6,64}$/;

const check = (id: string) => ID_RE.test(id);

const forward = async (
  method: "GET" | "PUT" | "DELETE",
  id: string,
  cookie: string,
  body?: string,
): Promise<NextResponse> => {
  if (!check(id)) return NextResponse.json({ error: "invalid id" }, { status: 400 });
  const headers: Record<string, string> = { ...backendOriginHeaders() };
  if (body) headers["Content-Type"] = "application/json";
  if (cookie) headers["cookie"] = cookie;
  const res = await fetch(`${pythonApiUrl()}/api/conversations/${encodeURIComponent(id)}`, {
    method,
    headers: Object.keys(headers).length ? headers : undefined,
    body,
    cache: "no-store",
  });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "Content-Type": res.headers.get("content-type") ?? "application/json" },
  });
};

async function handleGET(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  return forward("GET", id, req.headers.get("cookie") ?? "");
}

async function handlePUT(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const { id } = await ctx.params;
  const body = await req.text();
  return forward("PUT", id, req.headers.get("cookie") ?? "", body);
}

async function handleDELETE(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const { id } = await ctx.params;
  return forward("DELETE", id, req.headers.get("cookie") ?? "");
}

export const GET = publicRoute(handleGET);

export const PUT = publicRoute(handlePUT);

export const DELETE = publicRoute(handleDELETE);
