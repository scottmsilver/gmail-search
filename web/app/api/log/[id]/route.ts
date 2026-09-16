import { publicRoute } from "@/lib/publicBoundary";
import { NextRequest, NextResponse } from "next/server";
import { readFile } from "node:fs/promises";

import { logPathFor } from "@/lib/chatLog";

import { authenticatedUserId, unauthenticated, privateHeaders } from "@/lib/requestSecurity";

export const runtime = "nodejs";

const ID_RE = /^[a-f0-9]{6,32}$/;

async function handleGET(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const ownerId = await authenticatedUserId(req);
  if (!ownerId) return unauthenticated();
  const { id } = await ctx.params;
  if (!ID_RE.test(id)) {
    return NextResponse.json({ error: "invalid id" }, { status: 400, headers: privateHeaders });
  }
  try {
    const content = await readFile(logPathFor(id), "utf-8");
    const records = content.trim().split("\n").map(line => JSON.parse(line));
    if (!records.length || records.some(record => record.owner_id !== ownerId)) {
      return NextResponse.json({ error: "log not found" }, { status: 404, headers: privateHeaders });
    }
    return new NextResponse(content, {
      headers: { ...privateHeaders, "Content-Type": "application/x-ndjson", "Content-Disposition": "attachment" },
    });
  } catch {
    return NextResponse.json({ error: "log not found" }, { status: 404, headers: privateHeaders });
  }
}

export const GET = publicRoute(handleGET);
