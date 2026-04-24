// Per-million-token USD rates for Gemini models. Source of truth is
// kept parallel with `src/gmail_search/agents/cost.py` — update both
// when Google changes prices. We intentionally don't try to import
// the Python table at runtime; bun/node have no clean path to it and
// the two pipelines can drift by a day without anyone noticing.
//
// `cached_input` is stored for completeness but unused today —
// neither the AI SDK nor the Gemini grounded-search path exposes a
// cached-vs-uncached split in the usage object.

export type Pricing = {
  input: number;
  output: number;
  cached_input?: number;
};

export const GEMINI_PRICING: Record<string, Pricing> = {
  "gemini-2.5-pro": { input: 1.25, output: 10.0, cached_input: 0.3125 },
  "gemini-2.5-flash": { input: 0.075, output: 0.3 },
  "gemini-2.5-flash-lite": { input: 0.1, output: 0.4 },
  "gemini-3.1-pro-preview": { input: 1.25, output: 10.0 },
  "gemini-3-pro-preview": { input: 1.25, output: 10.0 },
  "gemini-3.1-flash-lite-preview": { input: 0.1, output: 0.4 },
  "gemini-3-flash-preview": { input: 0.075, output: 0.3 },
  // Fallback matches flash pricing so we never *undercount* by
  // accident when a brand-new model id shows up.
  default: { input: 0.075, output: 0.3 },
};

// Pick the pricing row that matches the model name. Exact match first,
// then longest-prefix match (so `gemini-2.5-flash-lite-preview-xyz`
// lands on `gemini-2.5-flash-lite`, not `gemini-2.5-flash`).
export const pricingForModel = (model: string): Pricing => {
  if (!model) return GEMINI_PRICING.default;
  if (GEMINI_PRICING[model]) return GEMINI_PRICING[model];
  const candidates = Object.keys(GEMINI_PRICING)
    .filter((k) => k !== "default" && model.startsWith(k))
    .sort((a, b) => b.length - a.length);
  if (candidates.length > 0) return GEMINI_PRICING[candidates[0]];
  return GEMINI_PRICING.default;
};

// Convert a (model, input, output) triple to a USD estimate. Returns
// 0 on zero tokens so a degenerate call that streamed nothing doesn't
// synthesize a phantom cost.
export const estimateCostUsd = (
  model: string,
  inputTokens: number,
  outputTokens: number,
): number => {
  if (inputTokens <= 0 && outputTokens <= 0) return 0;
  const p = pricingForModel(model);
  return (inputTokens * p.input) / 1_000_000 + (outputTokens * p.output) / 1_000_000;
};
