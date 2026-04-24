import type { AttachmentMetaFull } from "./backend";

// Mirrors threadCache.ts but keyed by attachment_id (number). Used by
// AttachmentChip to resolve a `[att:<id>]` citation into enough metadata
// to render a labeled chip and wire its click to the thread drawer.

type CacheEntry = AttachmentMetaFull | "loading" | "error";

const cache = new Map<number, CacheEntry>();
const subscribers = new Map<number, Set<() => void>>();

const notify = (id: number) => {
  subscribers.get(id)?.forEach((fn) => fn());
};

export const subscribeAttachment = (id: number, fn: () => void): (() => void) => {
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

export const getCachedAttachment = (id: number): CacheEntry | undefined => cache.get(id);

export const fetchAttachmentMeta = async (id: number): Promise<AttachmentMetaFull | null> => {
  const existing = cache.get(id);
  if (existing === "loading") return null;
  if (existing && existing !== "error") return existing;
  cache.set(id, "loading");
  notify(id);
  try {
    const res = await fetch(`/api/attachment/${encodeURIComponent(String(id))}/meta`);
    if (!res.ok) {
      cache.set(id, "error");
      notify(id);
      return null;
    }
    const data = (await res.json()) as AttachmentMetaFull;
    cache.set(id, data);
    notify(id);
    return data;
  } catch {
    cache.set(id, "error");
    notify(id);
    return null;
  }
};
