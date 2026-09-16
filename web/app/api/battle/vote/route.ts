import { publicRoute } from "@/lib/publicBoundary";
import { rejectUnsafeOrigin, backendOriginHeaders } from "@/lib/originSecurity";
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

async function handlePOST(req: NextRequest) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const body = await req.text();
  const upstream = await fetch(`${pythonApiUrl()}/api/battle/vote`, {
    method: "POST",
    headers: { ...backendOriginHeaders(), "Content-Type": "application/json", cookie: req.headers.get("cookie") ?? "" },
    body,
  });
  const data = await upstream.text();
  return new NextResponse(data, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export const POST = publicRoute(handlePOST);
