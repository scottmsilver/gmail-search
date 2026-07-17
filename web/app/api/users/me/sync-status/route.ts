// Initial-sync progress for the signed-in user. Cookie-forwarded so
// FastAPI's `require_user_id` resolves the session user.
import { NextRequest, NextResponse } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function GET(req: NextRequest) {
  const cookie = req.headers.get("cookie") ?? "";
  const upstream = await fetch(`${pythonApiUrl()}/api/users/me/sync-status`, {
    cache: "no-store",
    headers: cookie ? { cookie } : undefined,
  });
  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
