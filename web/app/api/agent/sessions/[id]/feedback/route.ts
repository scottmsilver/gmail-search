import { publicRoute } from "@/lib/publicBoundary";
import { rejectUnsafeOrigin, backendOriginHeaders } from "@/lib/originSecurity";
import { NextRequest, NextResponse } from "next/server";
import { pythonApiUrl } from "@/lib/config";

export const runtime = "nodejs";

async function handlePOST(
  req: NextRequest,
  context: { params: Promise<{ id: string }> },
) {
  const originDenied = rejectUnsafeOrigin(req);
  if (originDenied) return originDenied;
  const { id } = await context.params;
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(id)) {
    return NextResponse.json({ error: "Invalid session_id" }, { status: 400 });
  }
  const body = await req.text();
  const upstream = await fetch(
    `${pythonApiUrl()}/api/agent/sessions/${encodeURIComponent(id)}/feedback`,
    {
      method: "POST",
      headers: {
        ...backendOriginHeaders(),
        "Content-Type": "application/json",
        cookie: req.headers.get("cookie") ?? "",
      },
      body,
    },
  );
  const data = await upstream.text();
  return new NextResponse(data, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}

export const POST = publicRoute(handlePOST);
