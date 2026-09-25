// "What's new" release notes (#73). The deployer writes `whats-new.json` into
// each release's web directory (src/gmail_search/deploy/notes.py); the
// invited/public web serves it from /api/whats-new and shows it in a dialog.
// Every field is a commit subject or a number, rendered as React text only.

export type WhatsNewIssue = { number: number; pr: number | null; title: string };
export type WhatsNewRelease = { release: string; target: string; issues: WhatsNewIssue[] };
export type WhatsNewDoc = { releases: WhatsNewRelease[] };

export const WHATS_NEW_FILE = "whats-new.json";
export const WHATS_NEW_MAX_BYTES = 128 * 1024;
export const SEEN_RELEASE_KEY = "gms-whats-new-seen";

const cleanIssue = (value: unknown): WhatsNewIssue | null => {
  if (!value || typeof value !== "object") return null;
  const { number, pr, title } = value as Record<string, unknown>;
  if (!Number.isInteger(number) || typeof title !== "string") return null;
  return { number: number as number, pr: Number.isInteger(pr) ? (pr as number) : null, title };
};

const cleanRelease = (value: unknown): WhatsNewRelease | null => {
  if (!value || typeof value !== "object") return null;
  const { release, target, issues } = value as Record<string, unknown>;
  if (typeof release !== "string" || !Array.isArray(issues)) return null;
  return {
    release,
    target: typeof target === "string" ? target : "",
    issues: issues.map(cleanIssue).filter((i): i is WhatsNewIssue => i !== null),
  };
};

// Anything malformed degrades: a bad document is no releases, a bad entry is
// one fewer release or issue, never a thrown error.
export function sanitizeWhatsNew(value: unknown): WhatsNewDoc {
  const releases = value && typeof value === "object" ? (value as Record<string, unknown>).releases : null;
  if (!Array.isArray(releases)) return { releases: [] };
  return { releases: releases.map(cleanRelease).filter((r): r is WhatsNewRelease => r !== null) };
}

export function parseWhatsNew(text: string): WhatsNewDoc {
  if (text.length > WHATS_NEW_MAX_BYTES) return { releases: [] };
  try {
    return sanitizeWhatsNew(JSON.parse(text));
  } catch {
    return { releases: [] };
  }
}

export const latestRelease = (doc: WhatsNewDoc): string | null => doc.releases[0]?.release ?? null;

// Pop the dialog only when this browser recorded an earlier release: a
// first-ever visit has nothing to compare against and stays quiet.
export const isFreshRelease = (current: string | null, seen: string | null): boolean =>
  current !== null && seen !== null && seen !== current;

// Storage can be absent or throw (private mode, blocked site data). A blocked
// read looks like a first visit, so the popup never fires automatically.
export function readSeenRelease(storage: Pick<Storage, "getItem"> | undefined): string | null {
  try {
    return storage?.getItem(SEEN_RELEASE_KEY) ?? null;
  } catch {
    return null;
  }
}

export function writeSeenRelease(storage: Pick<Storage, "setItem"> | undefined, release: string): void {
  try {
    storage?.setItem(SEEN_RELEASE_KEY, release);
  } catch {
    // A failed write only means the popup may show again next load.
  }
}

// The body as text, or null past `cap` bytes: stops reading there, whether or
// not the response declared a length (a compressed one does not).
async function readCapped(response: Response, cap: number): Promise<string | null> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > cap) {
      void reader.cancel().catch(() => {});
      return null;
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(bytes);
}

// Same-origin, bounded: a slow or oversized response is "no notes".
export async function fetchWhatsNew(fetchImpl: typeof fetch = fetch): Promise<WhatsNewDoc> {
  try {
    const response = await fetchImpl("/api/whats-new", {
      cache: "no-store", credentials: "include", signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) return { releases: [] };
    const text = await readCapped(response, WHATS_NEW_MAX_BYTES);
    return text === null ? { releases: [] } : parseWhatsNew(text);
  } catch {
    return { releases: [] };
  }
}
