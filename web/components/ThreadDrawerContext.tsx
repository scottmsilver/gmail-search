"use client";

import { createContext, useCallback, useContext, useMemo, type ReactNode } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

type Ctx = {
  openThreadId: string | null;
  setOpenThreadId: (id: string | null) => void;
};

const ThreadDrawerCtx = createContext<Ctx | null>(null);

// Open-thread state lives in the URL (?thread=ID) so it survives reload
// and can be bookmarked / shared. Both /search and / (chat) read the same
// param, so the drawer state is consistent across tabs.
// Loose validation: thread IDs are Gmail's hex-ish 16-char strings. Reject
// anything else so a crafted ?thread=../etc/passwd never lands in a fetch URL.
const THREAD_ID_RE = /^[a-zA-Z0-9_-]{1,64}$/;

export const ThreadDrawerProvider = ({ children }: { children: ReactNode }) => {
  const router = useRouter();
  const pathname = usePathname() || "/";
  const params = useSearchParams();
  const rawThread = params?.get("thread") ?? null;
  const openThreadId = rawThread && THREAD_ID_RE.test(rawThread) ? rawThread : null;
  // Stringify once so callbacks don't re-create on each render and clobber
  // each other when two state writes land in the same tick.
  const paramsKey = params?.toString() ?? "";

  const setOpenThreadId = useCallback(
    (id: string | null) => {
      // Read live URL at click time — avoids stale-snapshot races when the
      // facet sidebar and drawer both write within the same render.
      const live = new URLSearchParams(typeof window !== "undefined" ? window.location.search : paramsKey);
      if (id && THREAD_ID_RE.test(id)) live.set("thread", id);
      else live.delete("thread");
      const qs = live.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [pathname, router, paramsKey],
  );

  const value = useMemo(() => ({ openThreadId, setOpenThreadId }), [openThreadId, setOpenThreadId]);
  return <ThreadDrawerCtx.Provider value={value}>{children}</ThreadDrawerCtx.Provider>;
};

export const useThreadDrawer = (): Ctx => {
  const ctx = useContext(ThreadDrawerCtx);
  if (!ctx) throw new Error("useThreadDrawer must be used inside ThreadDrawerProvider");
  return ctx;
};
