import { pythonApiUrl } from "./config";

// Cache the schema markdown in memory — it changes only when db.py
// changes, so a 1h TTL is more than safe. Re-fetch silently on TTL
// expiry; never fail the chat request if the schema endpoint is down.
let cached: { text: string; fetchedAt: number } | null = null;
let inflight: Promise<string> | null = null;
const TTL_MS = 60 * 60 * 1000;
const FETCH_TIMEOUT_MS = 3_000;

export const getSqlSchemaMarkdown = async (): Promise<string> => {
  if (cached && Date.now() - cached.fetchedAt < TTL_MS) return cached.text;
  // Coalesce concurrent cache-misses into one in-flight request so we
  // don't hammer the python server when many requests arrive together.
  if (inflight) return inflight;
  inflight = (async () => {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), FETCH_TIMEOUT_MS);
    try {
      const res = await fetch(`${pythonApiUrl()}/api/sql_schema`, {
        cache: "no-store",
        signal: ac.signal,
      });
      if (!res.ok) return cached?.text ?? "";
      const data = (await res.json()) as { markdown?: string };
      const text = data.markdown ?? "";
      cached = { text, fetchedAt: Date.now() };
      return text;
    } catch {
      return cached?.text ?? "";
    } finally {
      clearTimeout(timer);
      inflight = null;
    }
  })();
  return inflight;
};
