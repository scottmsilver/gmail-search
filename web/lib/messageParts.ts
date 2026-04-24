// Type guards for the assistant-ui message-part shapes we render in
// AssistantWork.tsx. Extracted into its own module — with NO React
// imports — so the unit test in `web/scripts/test-units.mjs` can
// import and exercise them without pulling in the assistant-ui React
// runtime.
//
// The DataMessagePart reshape (wire `data-<name>` → in-memory
// `{type: "data", name: "<name>"}`) is assistant-ui internal, not
// documented public API. The shape we rely on is `DataMessagePart`
// from `@assistant-ui/core` (see
// `node_modules/@assistant-ui/core/dist/types/message.d.ts`):
//
//   export type DataMessagePart<T = any> = {
//     readonly type: "data";
//     readonly name: string;
//     readonly data: T;
//   };
//
// If a future bump renames any of these fields the unit test that
// exercises `isDeepStagePart` will fail in CI before a user hits the
// silent breakage. The matching `@assistant-ui/react` and
// `@assistant-ui/react-ai-sdk` versions are pinned exactly in
// package.json (no caret) so `bun install` can't pull in a
// breaking minor.

export type ToolPart = {
  type: "tool-call";
  toolCallId: string;
  toolName: string;
  args: unknown;
  result?: unknown;
  isError?: boolean;
};

export type ReasoningPart = { type: "reasoning"; text: string };

export type DeepStagePart = {
  type: "data";
  name: "deep-stage";
  data: { kind: string; payload?: unknown };
};

export type WorkPart = ToolPart | ReasoningPart | DeepStagePart;

export const isToolPart = (p: { type: string }): p is ToolPart => p.type === "tool-call";

export const isReasoningPart = (p: { type: string }): p is ReasoningPart => p.type === "reasoning";

export const isDeepStagePart = (p: { type: string; name?: string }): p is DeepStagePart =>
  p.type === "data" && (p as { name?: string }).name === "deep-stage";
