import { mkdir, appendFile, readdir, stat, unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { randomBytes } from "node:crypto";

export const LOG_DIR =
  process.env.GMAIL_CHAT_LOG_DIR ?? path.join(homedir(), ".gmail-search-chat-logs");

// 8 bytes (16 hex chars) — 64 bits of entropy. Original 32-bit ID could
// collide at production volume; an 8-char filename is also too easy to
// brute-force enumerate via /api/log/[id] in any future deployment.
export const newRequestId = (): string => randomBytes(8).toString("hex");

export const logPathFor = (requestId: string): string =>
  path.join(LOG_DIR, `${requestId}.jsonl`);

const MAX_LOG_FILES = 500;

// Best-effort retention — keep only the N most-recently-modified log files.
// Runs on first write per process, never blocks the request.
let retentionRan = false;
const pruneOldLogs = async () => {
  if (retentionRan) return;
  retentionRan = true;
  try {
    const entries = await readdir(LOG_DIR);
    const jsonl = entries.filter((e) => e.endsWith(".jsonl"));
    if (jsonl.length <= MAX_LOG_FILES) return;
    const stats = await Promise.all(
      jsonl.map(async (f) => ({ f, mtime: (await stat(path.join(LOG_DIR, f))).mtimeMs })),
    );
    stats.sort((a, b) => b.mtime - a.mtime);
    const toDelete = stats.slice(MAX_LOG_FILES);
    for (const { f } of toDelete) {
      try {
        await unlink(path.join(LOG_DIR, f));
      } catch {
        // ignore — best effort
      }
    }
  } catch {
    // ignore — directory may not exist yet
  }
};

let dirEnsured = false;
const ensureDir = async () => {
  if (dirEnsured) return;
  if (!existsSync(LOG_DIR)) {
    await mkdir(LOG_DIR, { recursive: true });
  }
  dirEnsured = true;
  void pruneOldLogs();
};

const stripBinary = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(stripBinary);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (k === "base64" && typeof v === "string") {
        out[k] = `<${v.length} chars base64 stripped>`;
      } else {
        out[k] = stripBinary(v);
      }
    }
    return out;
  }
  return value;
};

export type LogEvent = {
  ts: string;
  kind:
    | "request"
    | "attempt"
    | "tool_call"
    | "tool_result"
    | "reasoning"
    | "answer"
    | "validation"
    | "error"
    | "done"
    | "battle_start"
    | "battle_done";
  data: Record<string, unknown>;
};

export class ChatLogger {
  readonly id: string;
  readonly path: string;
  private startMs: number;

  constructor(id?: string) {
    this.id = id ?? newRequestId();
    this.path = logPathFor(this.id);
    this.startMs = Date.now();
  }

  async log(kind: LogEvent["kind"], data: Record<string, unknown>): Promise<void> {
    await ensureDir();
    const event: LogEvent = {
      ts: new Date().toISOString(),
      kind,
      data: stripBinary(data) as Record<string, unknown>,
    };
    try {
      await appendFile(this.path, JSON.stringify(event) + "\n", "utf-8");
    } catch (err) {
      console.error(`[chatLog ${this.id}] write failed:`, err);
    }
  }

  elapsedMs(): number {
    return Date.now() - this.startMs;
  }
}
