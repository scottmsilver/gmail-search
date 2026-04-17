"use client";

/**
 * Holds the user's chosen model + thinking level for the active session.
 * The AssistantChatTransport `body` function reads from this on every
 * request, so changing settings takes effect on the NEXT message — no
 * need to recreate the runtime.
 *
 * Persisted to localStorage under a single key so reloads keep the
 * user's preference. All subscribers are notified on change.
 */
import { AVAILABLE_MODELS, DEFAULT_THINKING, THINKING_LEVELS, type ThinkingLevel } from "./config";

const STORAGE_KEY = "gmail-search-chat-settings-v1";

export type ChatSettings = {
  model: (typeof AVAILABLE_MODELS)[number];
  thinkingLevel: ThinkingLevel;
  battleMode: boolean;
};

const defaultSettings = (): ChatSettings => ({
  model: AVAILABLE_MODELS[0],
  thinkingLevel: DEFAULT_THINKING,
  battleMode: false,
});

let current: ChatSettings = defaultSettings();
let loaded = false;
const subscribers = new Set<() => void>();

const loadFromStorage = () => {
  if (loaded || typeof window === "undefined") return;
  loaded = true;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as Partial<ChatSettings>;
    const model = (AVAILABLE_MODELS as readonly string[]).includes(parsed.model ?? "")
      ? (parsed.model as ChatSettings["model"])
      : current.model;
    const thinkingLevel = (THINKING_LEVELS as string[]).includes(parsed.thinkingLevel ?? "")
      ? (parsed.thinkingLevel as ThinkingLevel)
      : current.thinkingLevel;
    const battleMode = typeof parsed.battleMode === "boolean" ? parsed.battleMode : current.battleMode;
    current = { model, thinkingLevel, battleMode };
  } catch {
    // ignore malformed storage
  }
};

export const getChatSettings = (): ChatSettings => {
  loadFromStorage();
  return current;
};

export const setChatSettings = (next: Partial<ChatSettings>) => {
  loadFromStorage();
  current = { ...current, ...next };
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(current));
  } catch {
    // ignore — private browsing, etc.
  }
  subscribers.forEach((fn) => fn());
};

export const subscribeChatSettings = (fn: () => void): (() => void) => {
  subscribers.add(fn);
  return () => {
    subscribers.delete(fn);
  };
};
