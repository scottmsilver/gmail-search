import { publicRoute } from "@/lib/publicBoundary";
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
// Don't cache — corpus stats change as the watch daemon syncs.
export const revalidate = 0;

async function handleGET(req: NextRequest) {
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/status`, {
    cache: "no-store",
    headers: cookie ? { cookie } : undefined,
  });
  const data = await upstream.text();
  return new NextResponse(data, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export const GET = publicRoute(handleGET);
