import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const upstream = await fetch(`${pythonApiUrl()}/api/thread/${encodeURIComponent(id)}`);
  if (!upstream.ok) {
    return NextResponse.json({ error: `upstream ${upstream.status}` }, { status: upstream.status });
  }
  const data = await upstream.json();
  return NextResponse.json(data);
}
