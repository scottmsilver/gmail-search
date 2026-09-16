import { publicRoute } from "@/lib/publicBoundary";
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

async function handleGET(req: NextRequest) {
  const upstream = await fetch(`${pythonApiUrl()}/api/battle/stats`, {headers: {cookie: req.headers.get("cookie") ?? ""}});
  const data = await upstream.text();
  return new NextResponse(data, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export const GET = publicRoute(handleGET);
