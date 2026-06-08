// Admin: stop the multi-user supervisor. Supervised daemons keep running.
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/admin/supervisor/stop`, {
    method: "POST",
    headers: cookie ? { cookie } : undefined,
    cache: "no-store",
  });
  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
