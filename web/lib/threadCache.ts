import type { ThreadDetail } from "./backend";

type CacheEntry = ThreadDetail | "loading" | "error";

const cache = new Map<string, CacheEntry>();
// Resolves a short cite_ref (or any prefix) to the full thread_id.
const refResolution = new Map<string, string | "miss">();
const subscribers = new Map<string, Set<() => void>>();

const notify = (id: string) => {
  subscribers.get(id)?.forEach((fn) => fn());
};

export const subscribeThread = (id: string, fn: () => void): (() => void) => {
  let set = subscribers.get(id);
  if (!set) {
    set = new Set();
    subscribers.set(id, set);
  }
  set.add(fn);
  return () => {
    set?.delete(fn);
  };
};

export const getCachedThread = (id: string): CacheEntry | undefined => {
  // If id is a partial ref that's been resolved, look up under the full id.
  const resolved = refResolution.get(id);
  if (resolved && resolved !== "miss") return cache.get(resolved);
  return cache.get(id);
};

const resolveRef = async (ref: string): Promise<string | null> => {
  const cached = refResolution.get(ref);
  if (cached === "miss") return null;
  if (cached) return cached;
  try {
    const res = await fetch(`/api/thread_lookup/${encodeURIComponent(ref)}`);
    if (!res.ok) {
      refResolution.set(ref, "miss");
      return null;
    }
    const data = (await res.json()) as { thread_id: string };
    refResolution.set(ref, data.thread_id);
    return data.thread_id;
  } catch {
    refResolution.set(ref, "miss");
    return null;
  }
};

export const fetchThread = async (idOrRef: string): Promise<ThreadDetail | null> => {
  // Always resolve first so short cite_refs map to the full thread_id.
  const full = idOrRef.length >= 16 ? idOrRef : (await resolveRef(idOrRef)) ?? idOrRef;
  refResolution.set(idOrRef, full);

  const existing = cache.get(full);
  if (existing === "loading") return null;
  if (existing && existing !== "error") return existing;
  cache.set(full, "loading");
  notify(idOrRef);
  notify(full);
  try {
    const res = await fetch(`/api/thread/${encodeURIComponent(full)}`);
    if (!res.ok) throw new Error(`thread fetch ${res.status}`);
    const data = (await res.json()) as ThreadDetail;
    cache.set(full, data);
    notify(idOrRef);
    notify(full);
    return data;
  } catch {
    cache.set(full, "error");
    notify(idOrRef);
    notify(full);
    return null;
  }
};
