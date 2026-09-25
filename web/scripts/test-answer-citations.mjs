// The invited guest's system prompt (MAIL_GUIDANCE) tells the model how to
// cite mail; this renderer turns those citations into clickable chips. #40:
// the guidance never asked for thread citations, so answers carried none.
// Read the citation forms straight out of the guest file, fill in synthetic
// IDs, and render them the way a stored answer is rendered, so a change on
// either side that stops claims showing their emails fails here.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import React, { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

globalThis.React = React; // components use the classic JSX runtime under tsx
const { CitableMarkdown } = await import("../components/CitableMarkdown.tsx");
const { ThreadDrawerCtx } = await import("../components/ThreadDrawerContext.tsx");

const GUEST = new URL("../../deploy/public/worker/guest_agent_pi.py", import.meta.url);
const THREAD_ID = "18c2f0a1b2c3d4e5";

const guidance = () => {
  const source = readFileSync(GUEST, "utf8");
  const block = source.match(/^MAIL_GUIDANCE=\(([\s\S]*?)\)\n\n/m);
  assert.ok(block, "MAIL_GUIDANCE not found in the guest runner");
  return block[1];
};

const render = (text) =>
  renderToStaticMarkup(
    createElement(
      ThreadDrawerCtx.Provider,
      { value: { openThreadId: null, setOpenThreadId() {} } },
      createElement(CitableMarkdown, { text, hints: [] }),
    ),
  );

test("the guest asks the model to cite each claim's thread", () => {
  const forms = guidance().match(/\[ref:[A-Z_]+\]/g) ?? [];
  assert.ok(forms.length > 0, "MAIL_GUIDANCE never asks for a [ref:...] thread citation");
});

test("an answer cited as the guest asks renders a clickable chip for the thread", () => {
  const [form] = guidance().match(/\[ref:[A-Z_]+\]/g) ?? [];
  assert.ok(form, "MAIL_GUIDANCE never asks for a [ref:...] thread citation");
  const answer = `The synthetic invoice was paid on 3 March ${form.replace(/:[A-Z_]+/, `:${THREAD_ID}`)}.`;
  const html = render(answer);
  assert.ok(!html.includes("[ref:"), "citation left as raw text");
  assert.match(html, new RegExp(`<button type="button"[^>]*title="${THREAD_ID.slice(0, 10)}"`));
});

test("an uncited claim renders no chip", () => {
  assert.ok(!render("The synthetic invoice was paid on 3 March.").includes("<button"));
});
