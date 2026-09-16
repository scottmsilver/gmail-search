export const pythonApiUrl = (): string => {
  const url = process.env.PYTHON_API_URL;
  if (!url) {
    throw new Error("PYTHON_API_URL is not set — see .env.local.example");
  }
  return url.replace(/\/$/, "");
};

// Picker choices when the deep-mode backend is set to Claude Code.
// These are the alias names accepted by the Claude Code runtime.
export const CLAUDE_AVAILABLE_MODELS = [
  "sonnet",
  "opus",
  "haiku",
  "opusplan",
] as const;

export const PI_AVAILABLE_MODELS = [
  "google/gemini-3.8-flash",
  "openrouter/meta/muse-spark-1.3",
  "anthropic/claude-opus-5",
] as const;

export const piModelForRequest = (model: unknown): string => {
  const aliases: Record<string, string> = {
    "openrouter/google/gemini-3.8-flash": "google/gemini-3.8-flash",
    "openrouter/anthropic/claude-opus-5": "anthropic/claude-opus-5",
  };
  const selected = typeof model === "string" ? aliases[model] ?? model : "";
  return (PI_AVAILABLE_MODELS as readonly string[]).includes(selected) ? selected : PI_AVAILABLE_MODELS[0];
};

export type DeepBackend = "claude_code" | "pi";

export const availableModelsFor = (backend: DeepBackend): readonly string[] =>
  backend === "pi" ? PI_AVAILABLE_MODELS : CLAUDE_AVAILABLE_MODELS;

// Retained for labels on historical battles.
export type ThinkingLevel = "minimal" | "low" | "medium" | "high";
