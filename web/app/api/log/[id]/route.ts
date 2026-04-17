import { NextRequest, NextResponse } from "next/server";
import { readFile } from "node:fs/promises";

import { logPathFor } from "@/lib/chatLog";

export const runtime = "nodejs";

const ID_RE = /^[a-f0-9]{6,32}$/;

export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  if (!ID_RE.test(id)) {
    return NextResponse.json({ error: "invalid id" }, { status: 400 });
  }
  try {
    const content = await readFile(logPathFor(id), "utf-8");
    return new NextResponse(content, {
      headers: { "Content-Type": "application/x-ndjson" },
    });
  } catch {
    return NextResponse.json({ error: "log not found" }, { status: 404 });
  }
}
