import { type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

// Proxy for /api/artifact/<id>. The Writer cites analyst-produced
// plots / CSVs as [art:<id>]; the UI resolves that chip by hitting
// this route, which forwards to the Python endpoint. Bytes flow
// straight through with the original content-type preserved so
// images render inline and CSVs download.
export const runtime = "nodejs";

export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }): Promise<Response> {
  const { id } = await ctx.params;
  const upstream = await fetch(`${pythonApiUrl()}/api/artifact/${encodeURIComponent(id)}`);
  if (!upstream.ok) {
    return new Response(`artifact ${id} not found`, { status: upstream.status });
  }
  const headers = new Headers();
  const ct = upstream.headers.get("content-type");
  if (ct) headers.set("Content-Type", ct);
  const cd = upstream.headers.get("content-disposition");
  if (cd) headers.set("Content-Disposition", cd);
  return new Response(upstream.body, { status: 200, headers });
}
