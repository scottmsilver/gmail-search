import { AVAILABLE_MODELS, THINKING_LEVELS, type ThinkingLevel } from "./config";

export type BattleVariant = {
  model: string;
  thinkingLevel: ThinkingLevel;
};

// Full Cartesian of models × thinking levels. For Gemini 2.x (which
// doesn't accept thinkingConfig at all) we include ONE entry with a
// placeholder thinkingLevel that the server just ignores. For 3.x
// models we include all four thinking levels so the battle can surface
// how much thinking actually matters.
const supportsThinking = (model: string) => model.startsWith("gemini-3");

export const BATTLE_VARIANTS: BattleVariant[] = AVAILABLE_MODELS.flatMap((model) =>
  supportsThinking(model)
    ? THINKING_LEVELS.map((lvl) => ({ model, thinkingLevel: lvl }))
    : [{ model, thinkingLevel: "low" as ThinkingLevel }],
);

export const variantLabel = (v: BattleVariant): string => {
  const short = v.model.replace("gemini-", "").replace("-preview", "");
  return supportsThinking(v.model) ? `${short} · ${v.thinkingLevel}` : short;
};

export const pickTwoRandomVariants = (): [BattleVariant, BattleVariant] => {
  const a = BATTLE_VARIANTS[Math.floor(Math.random() * BATTLE_VARIANTS.length)];
  // Draw until we get a DIFFERENT variant — otherwise a small pool would
  // occasionally battle a model against itself.
  let b = a;
  while (
    b.model === a.model &&
    b.thinkingLevel === a.thinkingLevel
  ) {
    b = BATTLE_VARIANTS[Math.floor(Math.random() * BATTLE_VARIANTS.length)];
  }
  return [a, b];
};
