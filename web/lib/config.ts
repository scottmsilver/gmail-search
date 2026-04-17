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
