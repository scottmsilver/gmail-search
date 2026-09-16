# Readable mail for Pi

Pi defaults to `google/gemini-3.8-flash` when no explicit model is selected.
`GMAIL_PI_MODEL` and valid saved UI selections still take precedence. The
historical workflow-eval CLI keeps its pinned model; use a process-local model
override when comparing Gemini runs.

Agent thread reads convert HTML locally to Markdown in `body_text`, using
BeautifulSoup without a model call, network request or database backfill.
Stored messages and the web mail viewer retain originals. Conversion retains
source-order text, table rows, quotations, image alt labels and bounded links,
while removing scripts, styles and tracking images. It is not a summary.

```json
{"thread_ids": ["thread-id"], "message_ids": ["message-id"]}
```

Omit `message_ids` to read all messages in the chosen threads. Each returned
message has `id`, `cite_ref`, `body_format`, `body_offset`, `body_limit`,
`body_total_chars` and `body_next_offset`. Readable mode adds `body_source`
(`html` or `text`). Plain text is preserved exactly when HTML is empty or fails
conversion. `thread_message_count` includes unselected messages;
`returned_message_count` counts selected messages.

The default limit is 20,000 characters per message (allowed: 1–100,000).
Continue with `message_ids=[message.id]` and `body_offset=body_next_offset`.
Offsets refer to characters in the chosen representation. Null means no remainder.

Use `body_format="raw"` to inspect original `body_text` and `body_html`.
HTML has separate `body_html_total_chars` and `body_html_next_offset`; either
continuation uses the same `body_offset` parameter, so the other field may be
repeated or exhausted. CSS layout, row/column spans and unsafe or long link
addresses are not preserved in Markdown. Links over 300 characters retain labels
only. Inspect raw content when these details matter or conversion is empty.

The SI scales investigation to the question, recommends batched selected reads,
removes mandatory per-call narration and SQL body slicing, and requires source
verification, coverage checks, bounded error recovery and valid citations.
Optional workflow tools remain optional.

## Validation and activation

Synthetic tests cover receipts, tables, quotations, fallback, malformed HTML,
raw reads, selection, tenant forwarding and exact pagination. A private sample
of seven real messages converted in 2–59 ms each. Two HTML-only receipts shrank
from approximately 169k to 8.4k and 52k to 18k characters, retaining checked
flight/date/confirmation anchors. These are character counts, not token counts
or an end-to-end speed claim.

Private benchmark and follow-up eval artifacts are under
`data/agent-evals/readable-mail-20260910/`; do not commit them. The follow-up uses
two earlier questions, one baseline-profile run each, Gemini 3.8 Flash with medium
thinking, an isolated MCP server and disposable Pi container. SI and conversion
change together; this is a directional smoke eval, not an ablation.

Activation requires Python agent and MCP tools services to load updated code,
and the web app to load the model defaults. No image rebuild or mailbox rewrite
is needed for conversion. Start a fresh conversation to evaluate the new SI.
Production services were not restarted for this change.

The two follow-up runs completed: purchases 171→126 seconds, 42→30 model turns,
$1.13→$0.64; travel 181→145 seconds, 63→50 turns, $1.39→$1.26. These are single
runs against a live corpus. Spot checks still found message-ID citations in the
purchase answer and a missed older booking for future travel. See the private
report before interpreting this as a quality improvement.
