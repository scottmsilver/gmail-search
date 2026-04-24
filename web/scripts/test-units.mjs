#!/usr/bin/env node
// Lightweight unit suite for web/lib helpers — run with `node web/scripts/test-units.mjs`.
// We use tsx to load TS modules directly. No vitest, no jest — keeps the
// dependency surface tiny.

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

let passed = 0;
let failed = 0;
const fails = [];

const test = (name, fn) => {
  try {
    fn();
    passed++;
    process.stdout.write(`  ✓ ${name}\n`);
  } catch (err) {
    failed++;
    fails.push({ name, err });
    process.stdout.write(`  ✗ ${name}\n    ${err.message}\n`);
  }
};

const eq = (actual, expected, label = "") => {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) throw new Error(`${label}\n      expected: ${e}\n      actual:   ${a}`);
};

const truthy = (cond, label) => {
  if (!cond) throw new Error(`${label} — expected truthy`);
};

// Use tsx/esm/api to import TypeScript modules at runtime. tsx wraps the
// module exports under `.default` for namespace ESM compatibility.
const { tsImport } = await import("tsx/esm/api");
const loadTs = async (relPath) => {
  const m = await tsImport(join(ROOT, relPath), import.meta.url);
  return m.default ?? m;
};
const { sanitizeConversation } = await loadTs("lib/sanitizeConversation.ts");

console.log("\nsanitizeConversation");

test("drops empty assistant string content", () => {
  const out = sanitizeConversation([
    { role: "user", content: "hi" },
    { role: "assistant", content: "" },
    { role: "user", content: "still hi?" },
  ]);
  // After dropping the empty assistant, the two user turns merge.
  eq(out.length, 1, "should collapse to a single merged user turn");
  eq(out[0].role, "user");
  eq(out[0].content, "hi\n\nstill hi?");
});

test("drops empty assistant array content", () => {
  const out = sanitizeConversation([
    { role: "user", content: "a" },
    { role: "assistant", content: [] },
    { role: "user", content: "b" },
  ]);
  eq(out.length, 1);
  eq(out[0].content, "a\n\nb");
});

test("preserves alternating user/assistant turns", () => {
  const conv = [
    { role: "user", content: "hi" },
    { role: "assistant", content: "hello" },
    { role: "user", content: "ok" },
  ];
  const out = sanitizeConversation(conv);
  eq(out.length, 3);
  eq(out[1].content, "hello");
});

test("merges two adjacent user string turns", () => {
  const out = sanitizeConversation([
    { role: "user", content: "first" },
    { role: "user", content: "second" },
  ]);
  eq(out.length, 1);
  eq(out[0].content, "first\n\nsecond");
});

test("merges two adjacent assistant array turns into one array", () => {
  const out = sanitizeConversation([
    { role: "user", content: "q" },
    { role: "assistant", content: [{ type: "text", text: "a" }] },
    { role: "assistant", content: [{ type: "tool-call", toolCallId: "x", toolName: "y", args: {} }] },
  ]);
  eq(out.length, 2);
  eq(out[1].content, [
    { type: "text", text: "a" },
    { type: "tool-call", toolCallId: "x", toolName: "y", args: {} },
  ]);
});

test("merges mixed string + array assistant turns by promoting string to part", () => {
  const out = sanitizeConversation([
    { role: "user", content: "q" },
    { role: "assistant", content: "thinking..." },
    { role: "assistant", content: [{ type: "tool-call", toolCallId: "1", toolName: "f", args: {} }] },
  ]);
  eq(out.length, 2);
  // After merge: text part + tool-call part.
  eq(out[1].content, [
    { type: "text", text: "thinking..." },
    { type: "tool-call", toolCallId: "1", toolName: "f", args: {} },
  ]);
});

test("never merges through a tool-result turn", () => {
  const out = sanitizeConversation([
    { role: "user", content: "q" },
    { role: "assistant", content: [{ type: "tool-call", toolCallId: "1", toolName: "f", args: {} }] },
    { role: "tool", content: [{ type: "tool-result", toolCallId: "1", toolName: "f", output: 42 }] },
    { role: "assistant", content: "answer" },
  ]);
  eq(out.length, 4);
});

test("does not merge two adjacent tool turns", () => {
  const out = sanitizeConversation([
    { role: "tool", content: [{ type: "tool-result", toolCallId: "1", toolName: "f", output: 1 }] },
    { role: "tool", content: [{ type: "tool-result", toolCallId: "2", toolName: "f", output: 2 }] },
  ]);
  eq(out.length, 2, "tool turns are paired with the preceding assistant tool-call and must stay distinct");
});

test("empty input returns empty output", () => {
  eq(sanitizeConversation([]), []);
});

test("only empty assistants returns empty output", () => {
  eq(sanitizeConversation([
    { role: "assistant", content: "" },
    { role: "assistant", content: [] },
  ]), []);
});

// ─── safeUrl (CitableMarkdown's XSS guard) ─────────────────────────
console.log("\nCitableMarkdown safeUrl");

const SAFE_SCHEMES = /^(?:https?|mailto|ref):/i;
const safeUrl = (url) => {
  if (!url) return "";
  if (!/^[a-z][a-z0-9+.-]*:/i.test(url)) return url;
  return SAFE_SCHEMES.test(url) ? url : "";
};

test("blocks javascript: scheme", () => eq(safeUrl("javascript:alert(1)"), ""));
test("blocks data: scheme", () => eq(safeUrl("data:text/html,<script>alert(1)</script>"), ""));
test("blocks vbscript: scheme", () => eq(safeUrl("vbscript:msgbox()"), ""));
test("blocks file: scheme", () => eq(safeUrl("file:///etc/passwd"), ""));
test("allows https://", () => eq(safeUrl("https://example.com"), "https://example.com"));
test("allows http://", () => eq(safeUrl("http://example.com"), "http://example.com"));
test("allows mailto:", () => eq(safeUrl("mailto:a@b.com"), "mailto:a@b.com"));
test("allows ref://", () => eq(safeUrl("ref:abc123"), "ref:abc123"));
test("allows relative URLs (no scheme)", () => eq(safeUrl("/foo/bar"), "/foo/bar"));
test("allows fragment-only URLs", () => eq(safeUrl("#section"), "#section"));
test("blocks JaVaScRiPt: case-insensitive", () => eq(safeUrl("JaVaScRiPt:alert(1)"), ""));
test("empty string returns empty", () => eq(safeUrl(""), ""));

// ─── linkifyRefs ───────────────────────────────────────────────────
console.log("\nlinkifyRefs");

const { linkifyRefs, REF_PREFIX } = await loadTs("lib/linkifyRefs.ts");

test("prefix-resolves a short [ref:ID] to the full known thread_id", () => {
  // The model often emits an 8-char prefix; we resolve to the full id.
  const out = linkifyRefs("see [ref:abc12345]", ["abc12345abcd"]);
  truthy(
    out.includes(`[abc12345abcd](${REF_PREFIX}abc12345abcd)`),
    `got: ${out}`,
  );
});

test("emits a chip for bracketed unknown refs (lazy fetch)", () => {
  // Per the comment in linkifyRefs.ts: bracketed-but-unknown still becomes a
  // chip so the citation chip component can fetch it lazily.
  const out = linkifyRefs("see [ref:fake0000]", ["abc12345abcd"]);
  truthy(out.includes(`${REF_PREFIX}fake0000`), `got: ${out}`);
});

test("tolerates whitespace around ref id", () => {
  const out = linkifyRefs("see [ ref: abc12345 ]", ["abc12345abcd"]);
  truthy(out.includes(REF_PREFIX), `got: ${out}`);
});

test("ambiguous prefix (matches >1 known id) is not linkified to a single one", () => {
  // Two known IDs with the same prefix — resolveAgainstKnown returns null,
  // so the bracketed form falls through to the chip-emit branch.
  const out = linkifyRefs("see [ref:abc1]", ["abc12345abcd", "abc1234defgh"]);
  // It still emits a chip with the short form (lazy-fetch path).
  truthy(out.includes(`${REF_PREFIX}abc1`), `got: ${out}`);
});

test("bare unknown hex id is left alone (not turned into a citation)", () => {
  const out = linkifyRefs("commit deadbeef0000 was buggy", ["abc12345abcd"]);
  eq(out.includes(REF_PREFIX), false);
});

test("bare known hex id is linkified as a citation", () => {
  const out = linkifyRefs("see abc12345abcd", ["abc12345abcd"]);
  truthy(out.includes(`${REF_PREFIX}abc12345abcd`), `got: ${out}`);
});

// ─── sanitizeBodyExcerpt — tested via the same regex shape ─────────
console.log("\nbody-excerpt sanitization (anti-injection)");

const sanitizeBodyExcerpt = (text) =>
  text.replace(/[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]/g, "");

test("strips zero-width space", () =>
  eq(sanitizeBodyExcerpt("hi\u200Bthere"), "hithere"));
test("strips zero-width joiner", () =>
  eq(sanitizeBodyExcerpt("a\u200Db"), "ab"));
test("strips RTL override (U+202E)", () =>
  eq(sanitizeBodyExcerpt("safe\u202Edrows"), "safedrows"));
test("strips BOM (U+FEFF)", () =>
  eq(sanitizeBodyExcerpt("\uFEFFhello"), "hello"));
test("preserves normal whitespace", () =>
  eq(sanitizeBodyExcerpt("hi there\nfriend"), "hi there\nfriend"));
test("preserves accented characters", () =>
  eq(sanitizeBodyExcerpt("naïve résumé"), "naïve résumé"));

// ─── cleanSender (RFC 5322 wrapper) ────────────────────────────────
console.log("\ncleanSender");

const { cleanSender } = await loadTs("lib/sender.ts");

test("extracts display name from quoted form", () =>
  eq(cleanSender('"Scott Silver" <scott@example.com>'), "Scott Silver"));
test("extracts display name from unquoted form", () =>
  eq(cleanSender("Scott Silver <scott@example.com>"), "Scott Silver"));
test("returns email when only angle-bracketed address present", () =>
  eq(cleanSender("<connie.lin@morganstanley.com>"), "connie.lin@morganstanley.com"));
test("returns bare email as-is", () =>
  eq(cleanSender("plain@example.com"), "plain@example.com"));
test("handles empty input", () => eq(cleanSender(""), ""));
test("strips surrounding whitespace", () =>
  eq(cleanSender("  Alice <a@b.com>  "), "Alice"));
test("credentials after comma survive", () =>
  eq(cleanSender("Sasha Torres, MA, BCBA <sasha@x.com>"), "Sasha Torres, MA, BCBA"));
test("paren-vendor suffix stays in name", () =>
  eq(cleanSender('"San Bruno Flower Fashions (via X)" <x@y.com>'), "San Bruno Flower Fashions (via X)"));

// ─── isDeepStagePart reshape (assistant-ui DataMessagePart) ─────────
//
// Deep-mode stage events leave the server as `data-deep-stage` SSE
// frames. assistant-ui's client reshapes them in-memory into
// `{type: "data", name: "deep-stage", data: {...}}` (see
// node_modules/@assistant-ui/core/src/types/message.ts ->
// DataMessagePart). The AssistantWork component filters on the
// reshaped form. If a future @assistant-ui bump changes the field
// names ("name" -> "kind", say) this guard returns false silently,
// and the deep-mode disclosure stops rendering. This test catches
// that regression in CI before a user hits it.
console.log("\nisDeepStagePart reshape (assistant-ui DataMessagePart)");

const { isDeepStagePart, isToolPart, isReasoningPart } = await loadTs("lib/messageParts.ts");

test("matches a reshaped data-deep-stage part", () => {
  // This is the shape assistant-ui places into m.content after
  // consuming a `data-deep-stage` SSE frame from /api/agent/analyze.
  const part = {
    type: "data",
    name: "deep-stage",
    data: { kind: "plan", payload: { plan: { question_type: "factual" } } },
  };
  truthy(isDeepStagePart(part), "should accept a reshaped deep-stage part");
});

test("rejects a different data-* part (e.g. data-debug-id)", () => {
  // Other custom data parts share the `type: "data"` envelope; we
  // must NOT eat them or the wrong block renders for them.
  const part = { type: "data", name: "debug-id", data: { id: "abc" } };
  eq(isDeepStagePart(part), false, "data parts with a different name must not match");
});

test("rejects a tool-call part", () => {
  const part = { type: "tool-call", toolCallId: "1", toolName: "f", args: {} };
  eq(isDeepStagePart(part), false);
});

test("rejects a reasoning part", () => {
  eq(isDeepStagePart({ type: "reasoning", text: "thinking" }), false);
});

test("rejects the wire-format type 'data-deep-stage' (must be the reshaped form)", () => {
  // Catches the regression we hit once: code that filters on the
  // wire-format type instead of the reshaped {type:"data",
  // name:"deep-stage"} sees this part and silently drops it.
  const part = { type: "data-deep-stage", data: { kind: "plan" } };
  eq(isDeepStagePart(part), false);
});

test("rejects a data part missing the name field", () => {
  // Defensive: malformed parts mid-flight shouldn't trigger the
  // deep-stage branch.
  const part = { type: "data", data: { kind: "plan" } };
  eq(isDeepStagePart(part), false);
});

test("isToolPart and isReasoningPart still match their shapes (sanity)", () => {
  truthy(isToolPart({ type: "tool-call", toolCallId: "1", toolName: "f", args: {} }));
  truthy(isReasoningPart({ type: "reasoning", text: "x" }));
  eq(isToolPart({ type: "data", name: "deep-stage", data: {} }), false);
  eq(isReasoningPart({ type: "data", name: "deep-stage", data: {} }), false);
});

// ─── done ──────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
