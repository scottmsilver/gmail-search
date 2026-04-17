"use client";

import Link from "next/link";
import { useSyncExternalStore } from "react";

import {
  AVAILABLE_MODELS,
  THINKING_LEVELS,
  type ThinkingLevel,
} from "@/lib/config";
import {
  getChatSettings,
  setChatSettings,
  subscribeChatSettings,
  type ChatSettings,
} from "@/lib/chatSettings";

const useChatSettings = (): ChatSettings =>
  useSyncExternalStore(subscribeChatSettings, getChatSettings, getChatSettings);

// Short display names for the model dropdown.
const SHORT_NAME: Record<string, string> = {
  "gemini-3.1-flash-lite-preview": "3.1 Flash Lite",
  "gemini-3-pro-preview": "3 Pro",
  "gemini-2.5-pro": "2.5 Pro",
  "gemini-2.5-flash": "2.5 Flash",
  "gemini-2.5-flash-lite": "2.5 Flash Lite",
};

export const ModelPicker = () => {
  const settings = useChatSettings();

  return (
    <div className="flex items-center gap-3 text-xs text-neutral-500 px-4 pt-2">
      <div className="flex items-center gap-1.5">
        <span className="text-neutral-400">model</span>
        <select
          value={settings.model}
          onChange={(e) => setChatSettings({ model: e.target.value as ChatSettings["model"] })}
          className="bg-transparent text-neutral-700 hover:text-neutral-900 focus:outline-none cursor-pointer font-medium"
        >
          {AVAILABLE_MODELS.map((m) => (
            <option key={m} value={m}>
              {SHORT_NAME[m] ?? m}
            </option>
          ))}
        </select>
      </div>
      <div className="text-neutral-300">·</div>
      <div className="flex items-center gap-0.5 rounded-full bg-neutral-100 p-0.5">
        {THINKING_LEVELS.map((level) => (
          <button
            key={level}
            type="button"
            onClick={() => setChatSettings({ thinkingLevel: level as ThinkingLevel })}
            className={
              level === settings.thinkingLevel
                ? "px-2.5 py-0.5 rounded-full bg-white text-neutral-900 font-medium shadow-sm"
                : "px-2.5 py-0.5 rounded-full text-neutral-500 hover:text-neutral-800"
            }
            title={`thinking: ${level}`}
          >
            {level}
          </button>
        ))}
      </div>
      <Link
        href="/battle"
        className="ml-auto text-neutral-500 hover:text-neutral-800"
        title="A/B battle mode"
      >
        ⚔ battle
      </Link>
    </div>
  );
};
