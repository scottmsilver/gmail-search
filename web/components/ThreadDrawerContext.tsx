"use client";

import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

type Ctx = {
  openThreadId: string | null;
  setOpenThreadId: (id: string | null) => void;
};

const ThreadDrawerCtx = createContext<Ctx | null>(null);

export const ThreadDrawerProvider = ({ children }: { children: ReactNode }) => {
  const [openThreadId, setOpenThreadId] = useState<string | null>(null);
  const value = useMemo(() => ({ openThreadId, setOpenThreadId }), [openThreadId]);
  return <ThreadDrawerCtx.Provider value={value}>{children}</ThreadDrawerCtx.Provider>;
};

export const useThreadDrawer = (): Ctx => {
  const ctx = useContext(ThreadDrawerCtx);
  if (!ctx) throw new Error("useThreadDrawer must be used inside ThreadDrawerProvider");
  return ctx;
};
