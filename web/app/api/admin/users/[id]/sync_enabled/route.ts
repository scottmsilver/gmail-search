import { publicRoute } from "@/lib/publicBoundary";
import { backendOriginHeaders, rejectUnsafeOrigin } from "@/lib/originSecurity";
// Admin: enable/disable a user's auto-sync. Body: {enabled: boolean}.
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

const ID_RE = /^u_[a-zA-Z0-9_-]{6,32}$/;

async function handlePOST(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const { id } = await ctx.params;
  if (!ID_RE.test(id)) {
    return NextResponse.json({ error: "invalid user id" }, { status: 400 });
  }
  const cookie = req.headers.get("cookie") ?? "";
  const body = await req.text();
  const headers: Record<string, string> = { ...backendOriginHeaders(), "Content-Type": "application/json" };
  if (cookie) headers["cookie"] = cookie;
  const upstream = await fetch(
    `${pythonApiUrl()}/api/admin/users/${encodeURIComponent(id)}/sync_enabled`,
    { method: "POST", headers, body, cache: "no-store" },
  );
  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export const POST = publicRoute(handlePOST);
