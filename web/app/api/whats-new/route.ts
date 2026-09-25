// Serves the release notes the deployer wrote into this release's web
// directory (#73). Read at request time, so a controller-only release, which
// does not rebuild the web, still shows its entries. Invited/public web only:
// the owner web runs from the checkout and has no notes.

import { open } from "node:fs/promises";
import path from "node:path";

import { isPublicMode, publicRoute } from "@/lib/publicBoundary";
import { WHATS_NEW_FILE, WHATS_NEW_MAX_BYTES, parseWhatsNew, type WhatsNewDoc } from "@/lib/whatsNew";

export const runtime = "nodejs";
export const revalidate = 0;

async function readNotes(): Promise<WhatsNewDoc> {
  const file = path.join(process.cwd(), WHATS_NEW_FILE);
  // Reads at most one byte past the cap, so an oversized file (or one that
  // grows while being read) is refused without being buffered whole.
  const handle = await open(file, "r").catch(() => null);
  if (!handle) return { releases: [] };
  try {
    const buffer = Buffer.alloc(WHATS_NEW_MAX_BYTES + 1);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    if (bytesRead > WHATS_NEW_MAX_BYTES) return { releases: [] };
    return parseWhatsNew(buffer.subarray(0, bytesRead).toString("utf8"));
  } catch {
    return { releases: [] };
  } finally {
    await handle.close();
  }
}

async function handleGET() {
  if (!isPublicMode()) return Response.json({ error: "Not found" }, { status: 404 });
  return Response.json(await readNotes());
}

export const GET = publicRoute(handleGET);
