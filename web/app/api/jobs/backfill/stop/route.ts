import { publicRoute } from "@/lib/publicBoundary";
import { backendOriginHeaders, rejectUnsafeOrigin } from "@/lib/originSecurity";
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

async function handlePOST(req: NextRequest) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/jobs/backfill/stop`, {
    method: "POST",
    headers: { ...backendOriginHeaders(), ...(cookie ? { cookie } : {}) },
  });
  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export const POST = publicRoute(handlePOST);
