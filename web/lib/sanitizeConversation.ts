import type { ModelMessage } from "ai";

const isEmptyMessage = (m: ModelMessage): boolean => {
  const content = m.content;
  if (typeof content === "string") return content.trim().length === 0;
  if (Array.isArray(content)) return content.length === 0;
  return false;
};

// Coerce string content into the AI SDK's part-array shape so we can merge
// adjacent same-role turns even when one side is string and the other is
// already an array (e.g., assistant text + assistant tool-calls).
const toPartsArray = (content: unknown): Array<{ type: string; text?: string }> => {
  if (typeof content === "string") {
    return content.length > 0 ? [{ type: "text", text: content }] : [];
  }
  if (Array.isArray(content)) return content as Array<{ type: string }>;
  return [];
};

// Gemini APIs (especially 2.5 Flash) reject requests that contain empty
// assistant turns OR consecutive same-role turns ("finishReason: error"
// with no provider details). This happens when a prior battle variant
// returned empty and the UI persisted that empty turn. We drop empties
// and merge adjacent same-role turns to produce a clean alternating
// transcript that every provider accepts.
//
// Tool-result turns (role === "tool") must stay paired with the preceding
// assistant tool-call — we never merge through them.
export const sanitizeConversation = (messages: ModelMessage[]): ModelMessage[] => {
  const nonEmpty = messages.filter((m) => !(m.role === "assistant" && isEmptyMessage(m)));
  const merged: ModelMessage[] = [];
  for (const m of nonEmpty) {
    const prev = merged[merged.length - 1];
    if (prev && prev.role === m.role && m.role !== "tool" && prev.role !== "tool") {
      const prevParts = toPartsArray(prev.content);
      const curParts = toPartsArray(m.content);
      const combinedParts = [...prevParts, ...curParts];
      const allText = combinedParts.every((p) => p.type === "text");
      const newContent = allText
        ? combinedParts.map((p) => p.text ?? "").join("\n\n")
        : combinedParts;
      merged[merged.length - 1] = { ...prev, content: newContent } as ModelMessage;
      continue;
    }
    merged.push(m);
  }
  return merged;
};
