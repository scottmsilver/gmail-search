"use client";

import { useAuth } from "./AuthContext";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import {
  availableModelsFor,
  type DeepBackend,
} from "@/lib/config";
import {
  getChatSettings,
  getServerChatSettings,
  setChatSettings,
  subscribeChatSettings,
  THEMES,
  type ChatSettings,
  type Theme,
} from "@/lib/chatSettings";

const useChatSettings = (): ChatSettings =>
  useSyncExternalStore(subscribeChatSettings, getChatSettings, getServerChatSettings);

const SHORT_NAME: Record<string, string> = {
  "openrouter/meta/muse-spark-1.3": "Muse Spark 1.3 (OpenRouter)",
  "google/gemini-3.8-flash": "Gemini 3.8 Flash (Google)",
  "anthropic/claude-opus-5": "Claude Opus 5 (Anthropic)",
  sonnet: "Sonnet",
  opus: "Opus",
  haiku: "Haiku",
  opusplan: "Opus Plan",
};

const shortModel = (m: string) => SHORT_NAME[m] ?? m;

export const ModelPicker = () => {
  const { publicMode, fullRuntime } = useAuth();
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

  // Capability, not deployment: an allowlisted owner gets the full picker on
  // the public origin, and everyone else keeps the bounded retrieval loop.
  if (publicMode && !fullRuntime) return <span className="text-xs text-muted-foreground" title="Gemini searches your Gmail archive with retrieval tools. Shell execution and model battles are unavailable.">Gemini · Gmail retrieval only</span>;

  const battleOn = settings.battleMode;
  const deepBackend = settings.deepBackend;
  const modelOptions = availableModelsFor(deepBackend);
  const triggerLabel = battleOn
    ? "⚔ deep analysis battle"
    : `${deepBackend} · ${shortModel(settings.model)}`;

  const switchDeepBackend = (next: DeepBackend) => {
    const nextModels = availableModelsFor(next);
    const patch: Partial<ChatSettings> = { deepBackend: next };
    if (!(nextModels as readonly string[]).includes(settings.model)) {
      patch.model = nextModels[0] as ChatSettings["model"];
    }
    setChatSettings(patch);
  };

  return (
    <div ref={rootRef} className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-[11px] text-neutral-400 hover:text-neutral-700 font-mono px-1.5 py-0.5 rounded hover:bg-neutral-100 transition-colors whitespace-nowrap"
        title="Change analysis backend, model, or battle mode"
      >
        {triggerLabel} <span className="ml-0.5 opacity-60">▾</span>
      </button>

      {open && (
        <div className="absolute bottom-full left-0 mb-2 z-10 rounded-lg border border-neutral-200 bg-white shadow-lg p-3 min-w-[260px] text-xs text-neutral-600 flex flex-col gap-3">
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
              {modelOptions.map((m) => (
                <option key={m} value={m}>
                  {shortModel(m)}
                </option>
              ))}
            </select>
          </div>

          {(
            <div>
              <div className="text-[10px] uppercase tracking-wide text-neutral-400 mb-1">
                Deep backend
              </div>
              <div className="flex items-center gap-1">

                <button
                  type="button"
                  onClick={() => switchDeepBackend("claude_code")}
                  className={
                    deepBackend === "claude_code"
                      ? "flex-1 rounded bg-blue-700 text-white px-2 py-1 font-medium hover:bg-blue-600"
                      : "flex-1 rounded bg-neutral-100 text-neutral-700 px-2 py-1 hover:bg-neutral-200"
                  }
                  title="Claude Code runtime: Claudebox + MCP, full orchestrator (planner → ... → critic)."
                >
                  Claude Code
                </button>

                <button
                  type="button"
                  onClick={() => switchDeepBackend("pi")}
                  className={
                    deepBackend === "pi"
                      ? "flex-1 rounded bg-teal-700 text-white px-2 py-1 font-medium hover:bg-teal-600"
                      : "flex-1 rounded bg-neutral-100 text-neutral-700 px-2 py-1 hover:bg-neutral-200"
                  }
                  title="Pi: single agent using the selected model."
                >
                  Pi
                </button>
              </div>
            </div>
          )}

          <div>
            <div className="text-[10px] uppercase tracking-wide text-neutral-400 mb-1">
              Deep analysis battle
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
              ⚔ {battleOn ? "on — two independent deep analyses" : "off — one deep analysis"}
            </button>
          </div>

          <div>
            <div className="text-[10px] uppercase tracking-wide text-neutral-400 mb-1">
              Theme
            </div>
            <div className="grid grid-cols-2 gap-1">
              {THEMES.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setChatSettings({ theme: t as Theme })}
                  className={
                    t === settings.theme
                      ? "rounded px-2 py-1 bg-neutral-900 text-white text-[11px] font-medium capitalize"
                      : "rounded px-2 py-1 bg-neutral-100 text-neutral-700 hover:bg-neutral-200 text-[11px] capitalize"
                  }
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
