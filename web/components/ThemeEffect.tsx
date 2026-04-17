"use client";

import { useEffect, useSyncExternalStore } from "react";

import {
  getChatSettings,
  getServerChatSettings,
  subscribeChatSettings,
} from "@/lib/chatSettings";

// Tiny helper component: subscribes to chatSettings and keeps the
// <html data-theme="..."> attribute in sync. Rendered once near the
// root. On SSR it reads the server snapshot ("light"), then swaps to
// the user's stored theme after hydration — brief flash is acceptable
// for a local dev tool.
export const ThemeEffect = () => {
  const settings = useSyncExternalStore(
    subscribeChatSettings,
    getChatSettings,
    getServerChatSettings,
  );
  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.dataset.theme = settings.theme;
    }
  }, [settings.theme]);
  return null;
};
