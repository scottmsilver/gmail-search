import { PI_AVAILABLE_MODELS, type DeepBackend, type ThinkingLevel } from "./config";

export type BattleVariant = {
  // Optional for saved battles from before deep-only mode.
  backend?: DeepBackend | "adk" | "claude_native";
  model: string;
  thinkingLevel?: ThinkingLevel;
};

export const BATTLE_VARIANTS: BattleVariant[] = [
  ...["sonnet", "opus", "haiku"].map(model => ({ backend: "claude_code" as const, model })),
  ...PI_AVAILABLE_MODELS.map(model => ({ backend: "pi" as const, model })),
];

export const variantLabel = (v: BattleVariant): string => {
  const model = v.model.replace("gemini-", "").replace("-preview", "");
  return v.backend ? `${v.backend} · ${model}` : `${model}${v.thinkingLevel ? ` · ${v.thinkingLevel}` : ""}`;
};

export const pickTwoRandomVariants = (): [BattleVariant, BattleVariant] => {
  const index = Math.floor(Math.random() * BATTLE_VARIANTS.length);
  const offset = 1 + Math.floor(Math.random() * (BATTLE_VARIANTS.length - 1));
  return [BATTLE_VARIANTS[index], BATTLE_VARIANTS[(index + offset) % BATTLE_VARIANTS.length]];
};
