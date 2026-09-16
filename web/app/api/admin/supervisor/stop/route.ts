import { publicRoute } from "@/lib/publicBoundary";
import { backendOriginHeaders, rejectUnsafeOrigin } from "@/lib/originSecurity";
// Admin: stop the multi-user supervisor. Supervised daemons keep running.
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

async function handlePOST(req: NextRequest) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/admin/supervisor/stop`, {
    method: "POST",
    headers: { ...backendOriginHeaders(), ...(cookie ? { cookie } : {}) },
    cache: "no-store",
  });
  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export const POST = publicRoute(handlePOST);
