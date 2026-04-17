#!/usr/bin/env node
/**
 * test-chat-api.mjs — Battery tester for the Gmail chat API.
 *
 * POSTs AI SDK UIMessage requests to http://127.0.0.1:3001/api/chat,
 * parses the SSE stream, collects tool calls + answer text, and
 * validates that every [ref:X] and bare thread-id in the answer
 * resolves against known refs/thread_ids pulled from tool outputs.
 *
 * Usage:
 *   node scripts/test-chat-api.mjs                 # run full battery
 *   node scripts/test-chat-api.mjs --verbose       # dump raw events
 *   node scripts/test-chat-api.mjs --question "…"  # one-off question
 *
 * Exits 0 if all green, 1 if any test has unresolved refs.
 */

const ENDPOINT = process.env.CHAT_API_URL ?? "http://127.0.0.1:3001/api/chat";

const TEST_CASES = [
  "What is the status of the ADT security proposal?",
  "Tell me about my Form 8283 emails for crypto donations",
  "Find the most recent email from landmarks west and summarize it",
  "What did we decide about the roof?",
  "Who is the president of KE?",
  "Show me my newest email",
  "Summarize my construction project — team, workflow, areas",
  "What did Salvador say about the alarm system?",
];

const args = process.argv.slice(2);
const VERBOSE = args.includes("--verbose");
const qIdx = args.indexOf("--question");
const AD_HOC = qIdx >= 0 ? args[qIdx + 1] : null;

const BRACKET_RE = /\[\s*ref:\s*([a-zA-Z0-9_-]+)\s*\]/g;
const BARE_RE = /\b([a-f0-9]{14,18})\b/g;

function truncate(s, n) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function collectFields(node, key, out) {
  if (node == null) return;
  if (Array.isArray(node)) {
    for (const item of node) collectFields(item, key, out);
    return;
  }
  if (typeof node === "object") {
    for (const [k, v] of Object.entries(node)) {
      if (k === key && (typeof v === "string" || typeof v === "number")) {
        out.add(String(v));
      }
      collectFields(v, key, out);
    }
  }
}

function refResolves(ref, known) {
  if (known.has(ref)) return true;
  for (const k of known) if (k.startsWith(ref)) return true;
  return false;
}

async function runQuestion(question) {
  const body = {
    messages: [
      { id: "u1", role: "user", parts: [{ type: "text", text: question }] },
    ],
  };
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const toolInputs = [];
  const toolOutputs = [];
  let answer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of chunk.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        let ev;
        try {
          ev = JSON.parse(payload);
        } catch {
          continue;
        }
        if (VERBOSE) console.error("EV", ev.type ?? "?", ev);
        switch (ev.type) {
          case "text-delta":
            if (typeof ev.delta === "string") answer += ev.delta;
            else if (typeof ev.textDelta === "string") answer += ev.textDelta;
            break;
          case "tool-input-available":
            toolInputs.push(ev);
            break;
          case "tool-output-available":
            toolOutputs.push(ev);
            break;
          default:
            // ignore unknown / future event types gracefully
            break;
        }
      }
    }
  }

  const known = new Set();
  collectFields(toolOutputs, "cite_ref", known);
  collectFields(toolOutputs, "thread_id", known);

  const brackets = [...answer.matchAll(BRACKET_RE)].map((m) => m[1]);
  const bares = [...answer.matchAll(BARE_RE)]
    .map((m) => m[1])
    .filter((s) => /[a-f]/.test(s) && /[0-9]/.test(s));

  // Fallback: some tools (sql_query, get_attachment_text) don't expose
  // `thread_id` or `cite_ref` as named fields, they embed IDs in row
  // data. For any ref the model used that isn't in our known set, ask
  // the server to resolve it — matches the server's own validation.
  const unresolved = [...new Set([...brackets, ...bares])].filter((r) => !refResolves(r, known));
  await Promise.all(
    unresolved.map(async (ref) => {
      try {
        const res = await fetch(`${ENDPOINT.replace(/\/api\/chat$/, "")}/api/thread_lookup/${ref}`);
        if (res.ok) {
          const data = await res.json();
          if (data.thread_id) known.add(data.thread_id);
        }
      } catch {
        // ignore — treat as still unresolved
      }
    }),
  );

  const brokenBracket = brackets.filter((r) => !refResolves(r, known));
  const brokenBare = bares.filter((r) => !refResolves(r, known));

  return {
    question,
    toolCount: toolOutputs.length,
    knownCount: known.size,
    brokenBracket,
    brokenBare,
    answer,
  };
}

function printRow(r) {
  const ok = r.brokenBracket.length === 0 && r.brokenBare.length === 0;
  const mark = ok ? "\u2705" : "\u274C";
  const q = truncate(r.question, 60).padEnd(60);
  console.log(
    `${mark} ${q} tools=${String(r.toolCount).padStart(2)} ` +
      `broken-bracket=${r.brokenBracket.length} broken-bare=${r.brokenBare.length}`,
  );
  if (!ok && VERBOSE) {
    if (r.brokenBracket.length) console.log("   brackets:", r.brokenBracket);
    if (r.brokenBare.length) console.log("   bares:   ", r.brokenBare);
  }
  return ok;
}

const questions = AD_HOC ? [AD_HOC] : TEST_CASES;
const results = [];
for (const q of questions) {
  try {
    const r = await runQuestion(q);
    results.push(r);
    printRow(r);
  } catch (err) {
    console.log(`\u274C ${truncate(q, 60).padEnd(60)} ERROR: ${err.message}`);
    results.push({ question: q, brokenBracket: ["ERR"], brokenBare: [] });
  }
}

const passed = results.filter(
  (r) => r.brokenBracket.length === 0 && r.brokenBare.length === 0,
).length;
console.log(`\nSummary: ${passed}/${results.length} passed`);
process.exit(passed === results.length ? 0 : 1);
