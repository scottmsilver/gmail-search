import type { ThreadDetail } from "./backend";

type CacheEntry = ThreadDetail | "loading" | "error";

const cache = new Map<string, CacheEntry>();
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

export const getCachedThread = (id: string): CacheEntry | undefined => cache.get(id);

export const fetchThread = async (id: string): Promise<ThreadDetail | null> => {
  const existing = cache.get(id);
  if (existing && existing !== "error") {
    if (existing === "loading") return null;
    return existing;
  }
  cache.set(id, "loading");
  notify(id);
  try {
    const res = await fetch(`/api/thread/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(`thread fetch ${res.status}`);
    const data = (await res.json()) as ThreadDetail;
    cache.set(id, data);
    notify(id);
    return data;
  } catch {
    cache.set(id, "error");
    notify(id);
    return null;
  }
};
