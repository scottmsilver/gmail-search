"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import {
  AVAILABLE_MODELS,
  THINKING_LEVELS,
  type ThinkingLevel,
} from "@/lib/config";
import {
  getChatSettings,
  getServerChatSettings,
  setChatSettings,
  subscribeChatSettings,
  type ChatSettings,
} from "@/lib/chatSettings";

const useChatSettings = (): ChatSettings =>
  useSyncExternalStore(subscribeChatSettings, getChatSettings, getServerChatSettings);

const SHORT_NAME: Record<string, string> = {
  "gemini-3.1-pro-preview": "3.1 Pro",
  "gemini-3.1-flash-lite-preview": "3.1 Flash Lite",
  "gemini-3-pro-preview": "3 Pro",
  "gemini-3-flash-preview": "3 Flash",
  "gemini-2.5-pro": "2.5 Pro",
  "gemini-2.5-flash": "2.5 Flash",
  "gemini-2.5-flash-lite": "2.5 Flash Lite",
};

const shortModel = (m: string) => SHORT_NAME[m] ?? m;

export const ModelPicker = () => {
  const settings = useChatSettings();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  // Click outside → close
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    window.addEventListener("keydown", onEsc);
    return () => {
      window.removeEventListener("mousedown", onClick);
      window.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  const battleOn = settings.battleMode;
  const triggerLabel = battleOn
    ? "⚔ battle mode"
    : `${shortModel(settings.model)} · ${settings.thinkingLevel}`;

  return (
    <div ref={rootRef} className="relative flex justify-end px-4 pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-[11px] text-neutral-400 hover:text-neutral-700 font-mono px-2 py-0.5 rounded hover:bg-neutral-100 transition-colors"
        title="Change model / thinking / battle"
      >
        {triggerLabel} <span className="ml-0.5 opacity-60">▾</span>
      </button>

      {open && (
        <div className="absolute bottom-full right-4 mb-2 z-10 rounded-lg border border-neutral-200 bg-white shadow-lg p-3 min-w-[260px] text-xs text-neutral-600 flex flex-col gap-3">
          <div className={battleOn ? "opacity-40 pointer-events-none" : ""}>
            <div className="text-[10px] uppercase tracking-wide text-neutral-400 mb-1">
              Model
            </div>
            <select
              value={settings.model}
              onChange={(e) => setChatSettings({ model: e.target.value as ChatSettings["model"] })}
              disabled={battleOn}
              className="w-full bg-neutral-50 border border-neutral-200 rounded px-2 py-1 text-neutral-800 font-medium focus:outline-none focus:border-neutral-400"
            >
              {AVAILABLE_MODELS.map((m) => (
                <option key={m} value={m}>
                  {shortModel(m)}
                </option>
              ))}
            </select>
          </div>

          <div className={battleOn ? "opacity-40 pointer-events-none" : ""}>
            <div className="text-[10px] uppercase tracking-wide text-neutral-400 mb-1">
              Thinking
            </div>
            <div className="flex items-center gap-0.5 rounded-full bg-neutral-100 p-0.5">
              {THINKING_LEVELS.map((level) => (
                <button
                  key={level}
                  type="button"
                  disabled={battleOn}
                  onClick={() => setChatSettings({ thinkingLevel: level as ThinkingLevel })}
                  className={
                    level === settings.thinkingLevel
                      ? "flex-1 px-2 py-0.5 rounded-full bg-white text-neutral-900 font-medium shadow-sm"
                      : "flex-1 px-2 py-0.5 rounded-full text-neutral-500 hover:text-neutral-800"
                  }
                >
                  {level}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="text-[10px] uppercase tracking-wide text-neutral-400 mb-1">
              Battle mode
            </div>
            <button
              type="button"
              onClick={() => setChatSettings({ battleMode: !battleOn })}
              className={
                battleOn
                  ? "w-full rounded bg-neutral-900 text-white px-2 py-1 font-medium hover:bg-neutral-700"
                  : "w-full rounded bg-neutral-100 text-neutral-700 px-2 py-1 hover:bg-neutral-200"
              }
            >
              ⚔ {battleOn ? "on — two random variants per question" : "off — single model"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
