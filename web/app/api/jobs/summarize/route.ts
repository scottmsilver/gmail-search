import { NextResponse, type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function POST(req: NextRequest) {
  const url = new URL(`${pythonApiUrl()}/api/jobs/summarize`);
  for (const [k, v] of req.nextUrl.searchParams.entries()) {
    url.searchParams.set(k, v);
  }
  const upstream = await fetch(url.toString(), { method: "POST" });
  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
