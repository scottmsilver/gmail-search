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

// Picker choices shown in the UI. Add/remove here.
export const AVAILABLE_MODELS = [
  "gemini-3.1-flash-lite-preview",
  "gemini-3-pro-preview",
  "gemini-2.5-pro",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
] as const;

export type ThinkingLevel = "minimal" | "low" | "medium" | "high";
export const THINKING_LEVELS: ThinkingLevel[] = ["minimal", "low", "medium", "high"];

export const DEFAULT_THINKING: ThinkingLevel = "high";

export const isValidModel = (m: unknown): m is (typeof AVAILABLE_MODELS)[number] =>
  typeof m === "string" && (AVAILABLE_MODELS as readonly string[]).includes(m);

export const isValidThinking = (t: unknown): t is ThinkingLevel =>
  typeof t === "string" && (THINKING_LEVELS as string[]).includes(t);
