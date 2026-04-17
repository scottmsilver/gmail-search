import type { ThinkingLevel } from "./config";

export type BattleVariant = {
  model: string;
  thinkingLevel: ThinkingLevel;
};

// Keep this focused — fewer variants = more meaningful head-to-head
// counts for the same number of battles.
export const BATTLE_VARIANTS: BattleVariant[] = [
  { model: "gemini-3.1-flash-lite-preview", thinkingLevel: "low" },
  { model: "gemini-3.1-flash-lite-preview", thinkingLevel: "high" },
  { model: "gemini-2.5-flash", thinkingLevel: "low" },
  { model: "gemini-2.5-pro", thinkingLevel: "low" },
];

export const variantLabel = (v: BattleVariant): string =>
  `${v.model.replace("gemini-", "")} · ${v.thinkingLevel}`;

export const pickTwoRandomVariants = (): [BattleVariant, BattleVariant] => {
  const shuffled = [...BATTLE_VARIANTS].sort(() => Math.random() - 0.5);
  return [shuffled[0], shuffled[1]];
};
