import { NextResponse, type NextRequest } from "next/server";

import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";
export const revalidate = 0;

export async function GET(req: NextRequest) {
  const url = new URL(`${pythonApiUrl()}/api/search`);
  // Forward all incoming params verbatim — q, k, sort, filter, date_from/to.
  for (const [k, v] of req.nextUrl.searchParams.entries()) {
    url.searchParams.set(k, v);
  }
  const upstream = await fetch(url.toString(), { cache: "no-store" });
  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}
