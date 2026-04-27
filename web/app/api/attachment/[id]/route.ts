import { type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

// Streams the binary attachment from the Python backend so the browser
// only ever talks to one origin. Lets us serve everything behind one
// hostname (gms.i.oursilverfamily.com) without exposing the Python
// backend port directly.
export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const upstream = await fetch(`${pythonApiUrl()}/api/attachment/${encodeURIComponent(id)}`, {
    cache: "no-store",
  });
  const headers = new Headers();
  for (const h of ["content-type", "content-length", "content-disposition"]) {
    const v = upstream.headers.get(h);
    if (v) headers.set(h, v);
  }
  return new Response(upstream.body, { status: upstream.status, headers });
}
