import { NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
// Don't cache — corpus stats change as the watch daemon syncs.
export const revalidate = 0;

export async function GET() {
  const upstream = await fetch(`${pythonApiUrl()}/api/status`, { cache: "no-store" });
  const data = await upstream.text();
  return new NextResponse(data, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
