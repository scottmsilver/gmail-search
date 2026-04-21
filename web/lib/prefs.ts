"use client";

import { useEffect, useState } from "react";

// Client-only UI preferences backed by localStorage. Nothing goes to
// the server — these are purely rendering choices, and we want them
// to survive a refresh without adding a round-trip on every mount.

const SHOW_SEARCH_DEBUG_KEY = "gmail-search:show-search-debug";

const readBool = (key: string, fallback: boolean): boolean => {
  if (typeof window === "undefined") return fallback;
  const raw = window.localStorage.getItem(key);
  if (raw === null) return fallback;
  return raw === "1";
};

const writeBool = (key: string, value: boolean) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, value ? "1" : "0");
  // Notify other tabs + other components in THIS tab.
  window.dispatchEvent(new StorageEvent("storage", { key, newValue: value ? "1" : "0" }));
};

export const useShowSearchDebug = (): [boolean, (v: boolean) => void] => {
  const [value, setValue] = useState(() => readBool(SHOW_SEARCH_DEBUG_KEY, false));
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === SHOW_SEARCH_DEBUG_KEY) {
        setValue(readBool(SHOW_SEARCH_DEBUG_KEY, false));
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);
  const set = (v: boolean) => {
    writeBool(SHOW_SEARCH_DEBUG_KEY, v);
    setValue(v);
  };
  return [value, set];
};
