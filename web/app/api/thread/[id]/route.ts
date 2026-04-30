import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

export async function GET(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/thread/${encodeURIComponent(id)}`, {
    headers: cookie ? { cookie } : undefined,
    cache: "no-store",
  });
  if (!upstream.ok) {
    return NextResponse.json({ error: `upstream ${upstream.status}` }, { status: upstream.status });
  }
  const data = await upstream.json();
  return NextResponse.json(data);
}
