import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

export async function GET(req: NextRequest, ctx: { params: Promise<{ ref: string }> }) {
  const { ref } = await ctx.params;
  const url = new URL(`${pythonApiUrl()}/api/thread_lookup`);
  url.searchParams.set("cite_ref", ref);
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(url.toString(), {
    cache: "no-store",
    headers: cookie ? { cookie } : undefined,
  });
  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
