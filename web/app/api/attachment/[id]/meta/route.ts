import { NextResponse, type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const upstream = await fetch(`${pythonApiUrl()}/api/attachment/${encodeURIComponent(id)}/meta`, {
    cache: "no-store",
  });
  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}
