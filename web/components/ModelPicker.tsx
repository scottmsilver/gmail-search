"use client";

import { useSyncExternalStore } from "react";

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

// Server snapshot must NOT read localStorage — otherwise SSR HTML
// (defaults) doesn't match the first client render (localStorage).
const useChatSettings = (): ChatSettings =>
  useSyncExternalStore(subscribeChatSettings, getChatSettings, getServerChatSettings);

// Short display names for the model dropdown.
const SHORT_NAME: Record<string, string> = {
  "gemini-3.1-pro-preview": "3.1 Pro",
  "gemini-3.1-flash-lite-preview": "3.1 Flash Lite",
  "gemini-3-pro-preview": "3 Pro",
  "gemini-3-flash-preview": "3 Flash",
  "gemini-2.5-pro": "2.5 Pro",
  "gemini-2.5-flash": "2.5 Flash",
  "gemini-2.5-flash-lite": "2.5 Flash Lite",
};

export const ModelPicker = () => {
  const settings = useChatSettings();

  const battleOn = settings.battleMode;
  return (
    <div className="flex items-center gap-3 text-xs text-neutral-500 px-4 pt-2">
      <div className="flex items-center gap-1.5" style={battleOn ? { opacity: 0.35 } : undefined}>
        <span className="text-neutral-400">model</span>
        <select
          value={settings.model}
          onChange={(e) => setChatSettings({ model: e.target.value as ChatSettings["model"] })}
          disabled={battleOn}
          className="bg-transparent text-neutral-700 hover:text-neutral-900 focus:outline-none cursor-pointer font-medium disabled:cursor-not-allowed"
          title={battleOn ? "battle mode picks two random variants" : undefined}
        >
          {AVAILABLE_MODELS.map((m) => (
            <option key={m} value={m}>
              {SHORT_NAME[m] ?? m}
            </option>
          ))}
        </select>
      </div>
      <div className="text-neutral-300">·</div>
      <div className="flex items-center gap-0.5 rounded-full bg-neutral-100 p-0.5" style={battleOn ? { opacity: 0.35 } : undefined}>
        {THINKING_LEVELS.map((level) => (
          <button
            key={level}
            type="button"
            disabled={battleOn}
            onClick={() => setChatSettings({ thinkingLevel: level as ThinkingLevel })}
            className={
              level === settings.thinkingLevel
                ? "px-2.5 py-0.5 rounded-full bg-white text-neutral-900 font-medium shadow-sm"
                : "px-2.5 py-0.5 rounded-full text-neutral-500 hover:text-neutral-800 disabled:cursor-not-allowed"
            }
            title={`thinking: ${level}`}
          >
            {level}
          </button>
        ))}
      </div>
      <button
        type="button"
        onClick={() => setChatSettings({ battleMode: !battleOn })}
        className={
          battleOn
            ? "ml-auto rounded-full bg-neutral-900 text-white px-2.5 py-0.5 font-medium hover:bg-neutral-700"
            : "ml-auto rounded-full bg-neutral-100 text-neutral-600 px-2.5 py-0.5 hover:bg-neutral-200"
        }
        title="Battle mode: two random variants per message, vote on the better answer"
      >
        ⚔ battle {battleOn ? "on" : "off"}
      </button>
    </div>
  );
};
