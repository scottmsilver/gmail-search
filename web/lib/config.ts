export const pythonApiUrl = (): string => {
  const url = process.env.PYTHON_API_URL;
  if (!url) {
    throw new Error("PYTHON_API_URL is not set — see .env.local.example");
  }
  return url.replace(/\/$/, "");
};

export const geminiApiKey = (): string => {
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) {
    throw new Error("GEMINI_API_KEY is not set — see .env.local.example");
  }
  return key;
};

export const AGENT_MODEL = "gemini-3.1-flash-lite-preview";

// Picker choices shown in the UI. Sourced from the live Gemini API model
// list (v1beta). First entry is the default for new users.
export const AVAILABLE_MODELS = [
  "gemini-3.1-flash-lite-preview",
  "gemini-3.1-pro-preview",
  "gemini-3-pro-preview",
  "gemini-3-flash-preview",
  "gemini-2.5-pro",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
] as const;

// Note: "minimal" is valid in the SDK type but rejected by some Gemini 3.x
// models (e.g. 3.1-pro-preview responds "Thinking level MINIMAL is not
// supported for this model"). Dropping it from the UI + battle pool so
// we never hit that error silently. If a specific model needs it later,
// add a per-model capability map instead of this blanket list.
export type ThinkingLevel = "minimal" | "low" | "medium" | "high";
export const THINKING_LEVELS: ThinkingLevel[] = ["low", "medium", "high"];

export const DEFAULT_THINKING: ThinkingLevel = "high";

export const isValidModel = (m: unknown): m is (typeof AVAILABLE_MODELS)[number] =>
  typeof m === "string" && (AVAILABLE_MODELS as readonly string[]).includes(m);

export const isValidThinking = (t: unknown): t is ThinkingLevel =>
  typeof t === "string" && (THINKING_LEVELS as string[]).includes(t);
