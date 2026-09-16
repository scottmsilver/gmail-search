import { publicRoute } from "@/lib/publicBoundary";
import { backendOriginHeaders, rejectUnsafeOrigin } from "@/lib/originSecurity";
// Admin: stop a user's running frontfill/backfill/summarize daemon.
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

const ID_RE = /^u_[a-zA-Z0-9_-]{6,32}$/;
const JOB_KEYS = new Set(["frontfill", "backfill", "summarize"]);

async function handlePOST(
  req: NextRequest,
  ctx: { params: Promise<{ id: string; job_key: string }> },
) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const { id, job_key } = await ctx.params;
  if (!ID_RE.test(id)) return NextResponse.json({ error: "invalid user id" }, { status: 400 });
  if (!JOB_KEYS.has(job_key))
    return NextResponse.json({ error: "invalid job_key" }, { status: 400 });
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(
    `${pythonApiUrl()}/api/admin/users/${encodeURIComponent(id)}/${job_key}/stop`,
    { method: "POST", headers: { ...backendOriginHeaders(), ...(cookie ? { cookie } : {}) }, cache: "no-store" },
  );
  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export const POST = publicRoute(handlePOST);
