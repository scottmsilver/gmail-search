import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

const ID_RE = /^[a-zA-Z0-9_-]{6,64}$/;

const check = (id: string) => ID_RE.test(id);

const forward = async (
  method: "GET" | "PUT" | "DELETE",
  id: string,
  body?: string,
): Promise<NextResponse> => {
  if (!check(id)) return NextResponse.json({ error: "invalid id" }, { status: 400 });
  const res = await fetch(`${pythonApiUrl()}/api/conversations/${encodeURIComponent(id)}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body,
  });
  const text = await res.text();
  return new NextResponse(text, {
    status: res.status,
    headers: { "Content-Type": res.headers.get("content-type") ?? "application/json" },
  });
};

export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  return forward("GET", id);
}

export async function PUT(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const body = await req.text();
  return forward("PUT", id, body);
}

export async function DELETE(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  return forward("DELETE", id);
}
