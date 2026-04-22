"""Per-message summarization via a local Ollama model.

The summary is stored in `message_summaries` and surfaced in search
results so the agent doesn't need to fetch full thread bodies for the
common case.

Three-tier strategy:

1. `_auto_mail_summary` short-circuits clearly-low-value mail (promotions,
   social-network notifications) using Gmail's own CATEGORY_* labels plus
   sender-pattern fallbacks. No LLM call.
2. `_clean_body` strips URL spam, MIME headers, and Unicode preview
   artifacts before handing to the model — these were the top cause of
   model meltdowns under the previous qwen2.5 backend.
3. The LLM summarizes what remains. Default model is gemma4 (faster AND
   more accurate than qwen2.5:7b in benchmarks run 2026-04-19).
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import httpx

from gmail_search.llm import get_backend
from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)

# The storage key recorded in `message_summaries.model` combines the
# backend's model_id with a prompt version. Bumping PROMPT_VERSION (or
# swapping the backend) naturally triggers a re-summarize pass because
# old rows live under the old key — _messages_needing_summary only
# sees as "done" those rows matching the current composite key.
#
# We reuse the `model` column on purpose: adding a new column would
# need a migration, and the column is only consumed by two places
# (backfill dedup + search-time lookup), both of which route through
# this constant.
PROMPT_VERSION = "v6"
DEFAULT_MODEL = f"{get_backend().model_id}+{PROMPT_VERSION}"

# Budget math against vLLM's 8192-token context window (see
# gmail_search/llm/vllm.py). Measured ratio varies by content:
#   - "clean" prose: 0.30-0.35 tokens/char
#   - quoted-reply threads, HTML with URLs, base64-ish content:
#     up to 0.40-0.45 tokens/char (denser)
#
# Original cap was 20000 chars, which assumed 0.35 tok/char. Worked
# on the first run's recent messages, but the backfill's older
# mail (longer threads, more quoted history) hit 0.40+ and
# overflowed. Dropping to 15000 head + 4000 tail = 15000 chars max
# prompt body. Worst-case token math:
#   15000 chars × 0.45 tok/char = 6750 tokens body
#   + system prompt (~1600 chars = ~640 tokens)
#   + metadata (~110 chars = ~45 tokens)
#   + 500 output tokens
#   + 100 tokens safety
#   = ~8035 tokens — fits under 8192 with margin.
#
# Bench (scripts/bench_out/, 40 messages) validated the v5 prompt at
# this ballpark with 40/40 success and 3× specificity vs v1.
MAX_BODY_CHARS = 15000  # head cap before head+tail truncation
TAIL_CHARS = 4000  # for long bodies, keep this many from the end
SUMMARY_MAX_TOKENS = 500
HTTP_TIMEOUT = 120.0

# v5 from the bench: "name the sender first" drove most of the win
# — retrieval hits jump when the summary actually contains the
# person or org's name.
_SYSTEM_PROMPT = """You summarize emails for a retrieval index. Your output is used by
a search engine — specificity beats prose.

Output up to 4 sentences, under 600 characters total.

Always start with the sender's name or organization (e.g. "Stratechery
by Ben Thompson reports..." or "Rebecca Miller asks..."). Then capture
ALL specific facts: amounts, dates, names, places, decisions,
percentages, order/confirmation numbers, product/project names. Close
with any explicit ask (named: "X asks Y to do Z") or stop — no filler
like "no action required".

For long threads or newsletters, summarize the main content, not just
the opening paragraph. Every concrete number or proper noun in the
email should make it into the summary if it would help someone find
this email later by keyword.

Do NOT begin with "This email...", "The email...", "It looks like...",
"Based on...", "The message...", or similar preamble.

Do NOT wrap the output in quotes or prefix it with "Summary:", "TLDR:".

FORWARDED / SHARED LINKS: If the sender wrote little or nothing and the
email is mainly a link/attachment, say they SHARED it, not that they
announce/recommend it. If the linked content was captured (you'll see it
in the body or as an attachment), SUMMARIZE THE CONTENT itself — what
the article says, the recipe's key steps, the argument's thesis — not
just the title. Attribute the share to the sender at the start or end.
If the sender included a short personal note, include it verbatim in
quotes. If the linked content wasn't captured, say so plainly.

LINK URLS: If a `Links:` block is present in the user prompt, render
the most relevant URL as a Markdown link: `[short label](url)`. Always
close with `)` — never `]`. The label MUST be human-readable text
(article title, recipe name, call to action, product name) and must
be 2-6 words — NEVER put the URL itself inside the `[...]` brackets;
the URL belongs ONLY in the `(...)` parens. Put the link at the end
of the summary. Do not invent URLs; only copy from the `Links:` block.

PRESERVE CALL-TO-ACTION links always — "Pay now", "Confirm",
"Track package", "View statement", "RSVP", "Reset password",
"Download receipt", "Verify your email", "Complete your order",
"Review booking", "View invoice", "Open ticket". These are the links
the reader most needs to click later. For transactional mail these
links are more valuable than article links.

Only skip the `Links:` block entirely when every URL is clearly
noise (browser-preview of a newsletter, social-share buttons, signup
referrals) OR the email is a bare security code with no actionable
link.

Examples:
- "Rebecca Miller asks Rick Chen to confirm whether the kitchen's wood wall can move 12 inches right, noting the change shifts cabinet placement. She needs an answer by Friday so the contractor can finalize the order."
- "OpenAI billing charged $5.78 to card ending 9535 for API credit (order #ch_3N8X, April 18). No action required."
- "Stratechery (Ben Thompson) reports ESPN, Fox, and Warner Bros. Discovery are forming a joint sports-streaming service, each with one-third ownership, encompassing ~55% of U.S. sports rights and launching in the fall. Asks no action. [The real bundle](https://stratechery.com/2026/sports-bundle/)"
- "Alex Chen shared a link titled 'Why rent control backfires' with the note \\"worth a read\\" — linked content was not captured. [Why rent control backfires](https://example.com/rent-control)"
"""


# ─── auto-mail classifier ─────────────────────────────────────────────────


def _sender_display_name(from_addr: str) -> str:
    """`'"Alice Smith" <alice@example.com>'` → `Alice Smith`. Falls back
    to the bare email if there's no display name.
    """
    s = (from_addr or "").strip()
    m = re.match(r'^\s*"?([^"<]+?)"?\s*<', s)
    if m:
        return m.group(1).strip()
    # No angle brackets — just an email or plain string.
    return s.split("@")[0] if "@" in s else s or "unknown sender"


_NOREPLY_SENDER = re.compile(r"(^|<)(no[-_]?reply|donotreply|notification|updates|news|info)@", re.IGNORECASE)


def _auto_mail_summary(labels_json: str, from_addr: str) -> str | None:
    """Return a canonical short summary for clearly-automated mail so we
    skip the LLM entirely. Returns None when the message deserves a real
    summary — including CATEGORY_UPDATES (receipts, security codes) where
    the specific facts matter.
    """
    try:
        labels = set(json.loads(labels_json or "[]"))
    except (TypeError, ValueError):
        labels = set()
    sender = _sender_display_name(from_addr)

    if "CATEGORY_PROMOTIONS" in labels:
        return f"Promotional email from {sender}."
    if "CATEGORY_SOCIAL" in labels:
        return f"Social-network update from {sender}."
    # Note: we used to short-circuit CATEGORY_FORUMS too, but Gmail
    # applies FORUMS to *anything* sent to a group — including real
    # human-authored board updates, school bulletins, and investor
    # letters. Replacing those with a canned "Mailing-list post from
    # X" destroys retrievability. Now forums mail goes to the LLM
    # like any other; the GPU budget is cheap enough. See the
    # incident with msg 192cf297003c77e8 (2026-04-19).

    # Gmail didn't categorise but the sender screams "automated". We
    # only short-circuit here for senders that also sent zero signal
    # of being transactional. Keep this narrow — want OpenAI/Microsoft/
    # Capital One style notifications to reach the LLM so amounts/codes
    # get captured.
    # (Currently no non-label short-circuits — labels are the authority.)
    return None


# ─── body cleaning ────────────────────────────────────────────────────────

# U+034F (combining grapheme joiner) and related invisible chars that
# newsletters use as "preview line" filler.
_INVISIBLE_RUN = re.compile(r"[\u034f\u00ad\u200b-\u200f\ufeff]{2,}")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_MIME_HEADER = re.compile(r"^(Content-(Type|Transfer-Encoding|Disposition)):.*$", re.IGNORECASE | re.MULTILINE)
_BLANK_RUN = re.compile(r"\n\s*\n\s*\n+")


def _clean_body(body: str) -> str:
    """Strip noise that consistently tripped the previous summarizer:
    MIME headers, URL spam, invisible preview chars, giant blank gaps.
    """
    if not body:
        return ""
    s = body
    s = _MIME_HEADER.sub("", s)
    s = _INVISIBLE_RUN.sub("", s)
    s = _URL.sub("[link]", s)
    s = _BLANK_RUN.sub("\n\n", s)
    return s.strip()


def _truncate_body(body: str) -> str:
    """For bodies longer than MAX_BODY_CHARS, keep the first MAX_BODY_CHARS
    minus TAIL_CHARS plus the last TAIL_CHARS. Newsletters burn budget on
    boilerplate intros; this preserves any sign-off or call-to-action.
    """
    if len(body) <= MAX_BODY_CHARS:
        return body
    head_chars = MAX_BODY_CHARS - TAIL_CHARS
    return body[:head_chars] + "\n\n[...]\n\n" + body[-TAIL_CHARS:]


# Belt-and-suspenders budget check. If our char-based truncation
# under-estimated tokens/char (dense content), hard-cap the final
# prompt chars so we never send an over-budget request. Worst-case
# ratio we've measured is ~0.45 tok/char on quoted-reply threads.
#
#   budget tokens = 8192 - 500 output - 100 safety = 7592
#   budget chars  = 7592 / 0.45 ≈ 16870
#
# This is the TOTAL chars across (system + metadata + body) seen by
# the model. System prompt is ~1600 chars, so effective body+meta
# budget is ~15270 — matches the MAX_BODY_CHARS + TAIL_CHARS = 15000
# above with margin for metadata and trivia.
_PROMPT_HARD_CAP_CHARS = 16800


def _format_attachments(attachments: list[dict], budget_chars: int) -> str:
    """Render attachments into a prompt tail, packing as much extracted
    text as fits in `budget_chars`. Attachments with extracted text are
    included in descending order of text length (biggest first); each
    gets its fair share but never more than it actually has.

    Input shape: [{"filename": str, "extracted_text": str}, ...]
    """
    usable = [a for a in attachments if (a.get("extracted_text") or "").strip()]
    if not usable or budget_chars <= 200:
        return ""
    # Header costs ~30 chars per attachment. Reserve that out of budget.
    per_header = 30
    remaining = budget_chars - per_header * len(usable)
    if remaining <= 200:
        return ""
    # Split budget proportionally by available text length so small
    # attachments aren't truncated to nothing while a huge one hogs.
    usable.sort(key=lambda a: len(a["extracted_text"] or ""), reverse=True)
    parts: list[str] = []
    remaining_atts = len(usable)
    for a in usable:
        text = a.get("extracted_text") or ""
        share = remaining // max(remaining_atts, 1)
        # An attachment shouldn't take more than it has.
        take = min(share, len(text))
        if take < 80:  # too little to be useful — skip
            remaining_atts -= 1
            continue
        remaining -= take
        remaining_atts -= 1
        filename = a.get("filename") or "attachment"
        parts.append(f"\n\n[Attachment: {filename}]\n{text[:take]}")
    return "".join(parts)


# URL patterns we always want to drop before picking a "primary" link.
# Goal: a link the reader could click and get the thing the email is
# really about. Tracking, unsubscribe, and inline images are noise.
_LINK_NOISE_HINTS = (
    "unsubscribe",
    "preferences",
    "list-manage.com",
    "sparkpost",
    "sendgrid.net",
    "mail.beehiiv",
    "click.",
    "track.",
    "/unsub",
    "optout",
    "/px/",
    "pixel.",
    "view-in-browser",
    "viewonline",
)

_LINK_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp")


def _is_noisy_link(url: str) -> bool:
    low = url.lower()
    if any(low.endswith(ext) for ext in _LINK_IMAGE_EXTS):
        return True
    if any(hint in low for hint in _LINK_NOISE_HINTS):
        return True
    return False


def _extract_primary_links(body_text: str, limit: int = 2) -> list[str]:
    """Return up to `limit` distinct URLs from the raw body, in the
    order they appeared, skipping obvious noise (unsubscribe /
    tracking / preview images). Run BEFORE `_clean_body`, which
    replaces every URL with "[link]".
    """
    if not body_text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _URL.finditer(body_text):
        url = match.group(0).rstrip(".,);:>'\"")
        if url in seen:
            continue
        if _is_noisy_link(url):
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= limit:
            break
    return out


def _build_user_prompt(
    from_addr: str,
    subject: str,
    body_text: str,
    attachments: list[dict] | None = None,
) -> str:
    links = _extract_primary_links(body_text or "")
    body = _truncate_body(_clean_body(body_text or ""))
    prompt = f"From: {from_addr}\nSubject: {subject}\n\n{body}"
    if links:
        prompt += "\n\nLinks:\n" + "\n".join(links)

    # Pack as much attachment text as fits under the hard cap. Bodies
    # that hog the budget leave no room; short bodies (typical for
    # "See attached..." mail) leave 10K+ chars for the attachment.
    if attachments:
        remaining = _PROMPT_HARD_CAP_CHARS - len(prompt)
        att_text = _format_attachments(attachments, remaining)
        prompt += att_text

    # Last-line safety: if body + attachments are dense enough that
    # char-truncation still overshoots, chop more off while keeping
    # metadata. Prefers keeping the head (where the main content is).
    if len(prompt) > _PROMPT_HARD_CAP_CHARS:
        excess = len(prompt) - _PROMPT_HARD_CAP_CHARS
        prompt = prompt[: -excess - 3] + "..."
    return prompt


# ─── LLM call ─────────────────────────────────────────────────────────────


_STRIP_PREFIXES = ("Summary:", "SUMMARY:", "Subject:", "TLDR:", "TL;DR:")

# Gemma sometimes emits its chain-of-thought as the response content
# ("thought Thinking Process: 1. **Analyze the Request:**..."). This
# happens most on near-empty bodies (body==[link] after URL stripping)
# where the model has nothing to summarize. Drop the reasoning block
# and keep anything after it. If only reasoning was emitted, return
# empty so the caller can fail-fast.
# Opener matches either the bare "thought\n" token Gemma emits OR
# a line that begins with "Thinking Process:".
_THINK_OPENER = re.compile(r"^(thought\b|thinking process[:\s])", re.IGNORECASE)

# Gemma terminates reasoning in a few ways; we accept any of them as
# the handoff to the real answer. Each pattern is a non-greedy match
# that ends exactly where the real answer begins — do NOT use `[^\n]*`
# or you'll eat the answer along with the marker.
_THINK_END_PATTERNS = [
    # "(This is the response provided to the user.)" — Gemma's most
    # common closer. Matches "(this is the response...)" variants.
    re.compile(r"\(this is the[^)]*\)", re.IGNORECASE),
    # "**Final Output Generation.**" / "**Final Output.**" / "**Final Response:**"
    re.compile(r"\*\*final (?:output|response)(?: generation)?\.?\*\*", re.IGNORECASE),
    # Section-style handoff: a line that's just "Final answer:" or similar.
    re.compile(r"(?:^|\n)\s*(?:final answer|final output|summary|output|response)\s*[:\n]", re.IGNORECASE),
]


def _strip_thinking(s: str) -> str:
    """If the LLM emitted reasoning, try to return only the final answer
    that follows its end-of-thinking marker. If the response is pure
    reasoning with no usable answer, return empty so the caller can
    retry or mark failure — better than storing a chain-of-thought as
    a search summary.
    """
    if not _THINK_OPENER.match(s.strip()):
        return s
    last_end = -1
    for pat in _THINK_END_PATTERNS:
        for m in pat.finditer(s):
            if m.end() > last_end:
                last_end = m.end()
    if last_end >= 0:
        return s[last_end:].strip()
    return ""


def _clean_llm_output(raw: str) -> str:
    raw = raw.strip()
    raw = _strip_thinking(raw)
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()
    for prefix in _STRIP_PREFIXES:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix) :].strip()
    return raw


# ─── batched summarization ────────────────────────────────────────────────

_BATCH_SYSTEM_PROMPT = """You summarize emails for a retrieval index.

Input: multiple emails, each marked `--- email id=<id> ---`.

Output: ONE JSON object. Keys are the exact ids. Values are 1-2 sentence
summaries, each under 300 characters, capturing who sent it, specific
facts (amounts, dates, names, decisions), and any explicit ask.

Rules:
- Only include an ask when the sender explicitly requests an action. No
  filler like "no action required" or "no next step mentioned".
- Do not begin a summary with "This email...", "The email...", "It
  looks like...", "Based on...", etc.
- Output ONLY the JSON object. No prose around it, no markdown fencing.
"""


def _build_batch_user_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        body = _truncate_body(_clean_body(m["body_text"] or ""))
        parts.append(f"--- email id={m['id']} ---\nFrom: {m['from_addr']}\nSubject: {m['subject']}\n\n{body}")
    return "\n\n".join(parts)


def summarize_batch(
    client: httpx.Client,
    messages: list[dict],
    backend: Backend,
) -> dict[str, str]:
    """Summarize N messages in one LLM call, returning {id: summary}.

    Auto-classifiable mail (promotions/social/forums) bypasses the LLM
    entirely. The remaining messages go into a single prompt with
    JSON-object response formatting. If the response fails to parse or
    omits any id, the missing ids retry via summarize_one — so a bad
    batch never loses work, it just costs an extra call.

    Accepts dicts with keys: id, from_addr, subject, body_text,
    labels_json (optional, defaults to "[]").
    """
    if not messages:
        return {}

    out: dict[str, str] = {}
    llm_work: list[dict] = []
    for m in messages:
        auto = _auto_mail_summary(m.get("labels_json", "[]"), m["from_addr"])
        if auto:
            out[m["id"]] = auto
        else:
            llm_work.append(m)

    if not llm_work:
        return out

    parsed: dict = {}
    try:
        raw = backend.chat(
            client,
            messages=[
                {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": _build_batch_user_prompt(llm_work)},
            ],
            max_tokens=180 * len(llm_work),
            json_format=True,
        )
        obj = json.loads((raw or "").strip())
        if isinstance(obj, dict):
            parsed = obj
    except (httpx.HTTPError, ValueError):
        parsed = {}

    for m in llm_work:
        val = parsed.get(m["id"])
        if isinstance(val, str) and val.strip():
            out[m["id"]] = _clean_llm_output(val)

    # Per-email fallback for anything the batch didn't produce.
    missing = [m for m in llm_work if m["id"] not in out]
    for m in missing:
        try:
            s = summarize_one(
                client,
                from_addr=m["from_addr"],
                subject=m["subject"],
                body_text=m["body_text"],
                labels_json=m.get("labels_json", "[]"),
                backend=backend,
                attachments=m.get("attachments"),
            )
            if s:
                out[m["id"]] = s
        except Exception as e:
            logger.warning(f"per-email fallback failed for {m['id']}: {e!s}")

    return out


def summarize_one(
    client: httpx.Client,
    *,
    from_addr: str,
    subject: str,
    body_text: str,
    labels_json: str = "[]",
    backend: Backend,
    attachments: list[dict] | None = None,
) -> str:
    """Summarize one message. Auto-classify short-circuit first, LLM fallback.

    `attachments` is an optional list of {filename, extracted_text} dicts
    whose text is packed into the prompt tail. Essential for "See
    attached..." emails where the real content lives in a PDF/DOCX.
    """
    auto = _auto_mail_summary(labels_json, from_addr)
    if auto is not None:
        return auto

    raw = backend.chat(
        client,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(from_addr, subject, body_text, attachments)},
        ],
        max_tokens=SUMMARY_MAX_TOKENS,
    )
    return _clean_llm_output(raw)


def _fetch_attachments_for(conn, message_ids: list[str]) -> dict[str, list[dict]]:
    """One batched query returning `{message_id: [{filename, extracted_text}]}`
    for every attachment with non-trivial extracted text. Pulling in one
    query instead of N is the difference between a 100ms pending-list
    build and a 10s one at 20K messages.
    """
    if not message_ids:
        return {}
    placeholders = ",".join(["%s"] * len(message_ids))
    rows = conn.execute(
        f"""SELECT message_id, filename, extracted_text
            FROM attachments
            WHERE message_id IN ({placeholders})
              AND extracted_text IS NOT NULL
              AND length(extracted_text) > 80""",
        message_ids,
    ).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["message_id"], []).append({"filename": r["filename"], "extracted_text": r["extracted_text"]})
    return out


def _messages_needing_summary(conn, model: str, limit: int | None) -> list[dict]:
    """Return messages that don't yet have a summary under `model`, each
    with its attachments pre-joined. Attachments join happens in Python
    because SQL `GROUP_CONCAT` would force a column cap we'd have to
    special-case for long extracted text.
    """
    # Previously filtered `length(body_text) > 20` to avoid wasting LLM
    # calls on bounces / empty forwards. Dropped 2026-04-20 — empty-body
    # messages (delivery confirmations, calendar invites, LinkedIn "X
    # wants to connect") still have meaningful subject + from + labels,
    # and a 1-line summary of those is better than the UI showing "(no
    # summary yet)" forever. The backend prompt handles short inputs
    # gracefully.
    # Prioritize messages the user actually looks at: INBOX first, then
    # STARRED / IMPORTANT, then by recency. This matters a lot when
    # PROMPT_VERSION has just been bumped and the daemon is working
    # through ~173k existing rows — without this clause, the user's
    # inbox keeps showing old-prompt summaries for hours.
    # `labels` is a JSON-encoded array; LIKE is faster than parsing per
    # row and the false-positive rate is effectively zero (no legitimate
    # label contains another as a substring).
    sql = """
        SELECT m.id, m.from_addr, m.subject, m.body_text, m.body_html, m.labels
        FROM messages m
        LEFT JOIN message_summaries s
          ON s.message_id = m.id AND s.model = %s
        WHERE s.message_id IS NULL
        ORDER BY
          -- Frontfill wins over backfill: anything received in the
          -- last 24h (i.e. what `watch` / `sync_new_messages` just
          -- pulled) goes to the top of the queue, ahead of the
          -- 173k-message v6 re-backfill that's still grinding.
          (m.date::timestamptz > NOW() - INTERVAL '1 day') DESC,
          (m.labels LIKE '%%"INBOX"%%') DESC,
          (m.labels LIKE '%%"STARRED"%%' OR m.labels LIKE '%%"IMPORTANT"%%') DESC,
          m.date DESC
    """
    params: list = [model]
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    msg_ids = [r["id"] for r in rows]
    attachments_by_msg = _fetch_attachments_for(conn, msg_ids)
    from gmail_search.extract.text import html_to_text

    out: list[dict] = []
    for r in rows:
        # If body_text is empty but body_html is there, extract text
        # from the HTML so the summarizer has real content instead of
        # a blank body + just subject + sender (e.g. a lot of Gmail
        # marketing and transactional mail ships HTML-only; without
        # this fallback their summaries degenerate to "Confirms order"
        # with none of the useful details).
        body_text = (r["body_text"] or "").strip()
        if not body_text:
            body_html = (r["body_html"] or "").strip()
            if body_html:
                body_text = html_to_text(body_html)
        out.append(
            {
                "id": r["id"],
                "from_addr": r["from_addr"],
                "subject": r["subject"],
                "body_text": body_text,
                "labels": r["labels"],
                "attachments": attachments_by_msg.get(r["id"], []),
            }
        )
    return out


# Common LLM markdown-link mistakes that would otherwise render as a
# 900-char literal URL in the UI:
#   "[label](url]"   — wrong closing bracket
#   "[label](url"    — truncated before the close paren
# The second is trickier: we can't just append `)` because the URL
# itself may have been cut mid-token. Safer to drop the malformed
# link so it renders as plain sentence.
_MD_LINK_WRONG_CLOSE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s\]]*)\]")
_MD_LINK_TRUNCATED_TAIL = re.compile(r"\[([^\]]+)\]\((https?://[^)\s\]]*)\s*$")


def _repair_broken_markdown_links(text: str) -> str:
    """Fix the two malformed markdown-link patterns the Gemma summarizer
    emits occasionally. Called before the summary is stored so every
    consumer (UI, search, chat citations) sees clean markdown.
    """
    if not text:
        return text
    fixed = _MD_LINK_WRONG_CLOSE.sub(r"[\1](\2)", text)
    fixed = _MD_LINK_TRUNCATED_TAIL.sub(r"[\1](\2)", fixed)
    return fixed


def _store_summary(conn, message_id: str, summary: str, model: str) -> None:
    conn.execute(
        """INSERT INTO message_summaries (message_id, summary, model)
           VALUES (%s, %s, %s)
           ON CONFLICT(message_id) DO UPDATE SET
             summary = excluded.summary,
             model = excluded.model,
             created_at = CURRENT_TIMESTAMP""",
        (message_id, summary, model),
    )


def _record_summary_failure(conn, message_id: str, model: str, error: str) -> None:
    """Persist a summarization failure so it can be triaged later.

    Called from the summarizer's per-email and batch fallback paths
    whenever a message ends the cycle without a usable summary. The
    row is deleted in `_clear_summary_failure` on the next successful
    attempt, so the table is always "currently broken" — not an
    append-only log.
    """
    # Cap `error` to keep pathological backend tracebacks from
    # bloating the table; a 400-char head is enough to distinguish
    # known failure modes (context overflow, parse failures, backend
    # 5xx) from new ones.
    conn.execute(
        """INSERT INTO summary_failures (message_id, model, error, attempts)
           VALUES (%s, %s, %s, 1)
           ON CONFLICT (message_id) DO UPDATE SET
             model = EXCLUDED.model,
             error = EXCLUDED.error,
             attempts = summary_failures.attempts + 1,
             last_seen = NOW()""",
        (message_id, model, (error or "unknown")[:400]),
    )


def _clear_summary_failure(conn, message_id: str) -> None:
    conn.execute("DELETE FROM summary_failures WHERE message_id = %s", (message_id,))


def backfill(
    db_path: Path,
    *,
    concurrency: int = 12,
    batch_size: int = 1,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    """Summarize every message that doesn't yet have a summary under the
    active backend's model.

    The backend (Ollama / vLLM) is chosen by the LLM_BACKEND env var and
    owns both the model identity (recorded in message_summaries.model)
    and its own lifecycle — vLLM, for instance, spawns its server on
    enter and tears it down on exit.

    When `batch_size > 1`, messages are grouped into batches of that size
    and each batch is summarized in a single LLM call via summarize_batch
    (with per-email fallback for any ids the batch misses).

    Commits one message at a time so a crash mid-run doesn't lose work.
    """
    from gmail_search.store.db import JobProgress

    # Backend (Ollama / vLLM) chosen by env var. The `with backend:`
    # block owns lifecycle — e.g. vLLM spawns its own subprocess on
    # enter and tears it down on exit so the GPU isn't pinned between
    # jobs. The storage key combines backend model_id + PROMPT_VERSION
    # so a prompt change re-queues every message automatically.
    backend = get_backend()
    model = f"{backend.model_id}+{PROMPT_VERSION}"

    conn = get_connection(db_path)
    pending = _messages_needing_summary(conn, model, limit)
    total = len(pending)
    if total == 0:
        conn.close()
        return {"total": 0, "done": 0, "auto_classified": 0, "failed": 0, "seconds": 0.0}

    # Publish progress to job_progress so /api/jobs/running can surface
    # live rate + ETA for the /settings summarizer card. start_completed=0
    # because unlike backfill, this run starts from scratch (nothing of
    # the target already "done" at t=0).
    job = JobProgress(db_path, "summarize", start_completed=0)

    done = 0
    auto_classified = 0
    failed = 0
    start = time.time()

    def _persist(
        summaries_by_id: dict[str, str],
        batch_messages: list[dict],
        errors_by_id: dict[str, str] | None = None,
    ) -> None:
        nonlocal done, auto_classified, failed
        errs = errors_by_id or {}
        for m in batch_messages:
            summary = summaries_by_id.get(m["id"])
            if summary:
                summary = _repair_broken_markdown_links(summary)
                _store_summary(conn, m["id"], summary, model)
                _clear_summary_failure(conn, m["id"])
                done += 1
                if _auto_mail_summary(m["labels"], m["from_addr"]) is not None:
                    auto_classified += 1
            else:
                err = errs.get(m["id"]) or "missing_from_output"
                _record_summary_failure(conn, m["id"], model, err)
                failed += 1
        conn.commit()
        processed = done + failed
        job.update(
            "summarizing",
            processed,
            total,
            f"{done} ok · {auto_classified} auto · {failed} failed",
        )

    job.update("starting backend", 0, total, f"loading {backend.model_id}")
    try:
        with backend:
            try:
                with httpx.Client() as client:
                    if batch_size <= 1:
                        _run_per_email(client, pending, concurrency, backend, _persist, progress, start, total)
                    else:
                        _run_batched(
                            client, pending, concurrency, batch_size, backend, _persist, progress, start, total
                        )
            finally:
                job.finish(
                    "done",
                    f"{done}/{total} summarized ({auto_classified} auto, {failed} failed)",
                )
    except Exception as e:
        logger.error("backend failed: %s", e)
        job.finish("error", f"backend error: {e}")
        raise
    finally:
        conn.close()

    elapsed = time.time() - start
    return {
        "total": total,
        "done": done,
        "auto_classified": auto_classified,
        "failed": failed,
        "seconds": round(elapsed, 1),
    }


def _run_per_email(client, pending, concurrency, backend, persist, progress, start, total):
    last_log = start
    processed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                summarize_one,
                client,
                from_addr=m["from_addr"],
                subject=m["subject"],
                body_text=m["body_text"],
                labels_json=m["labels"],
                backend=backend,
                attachments=m.get("attachments"),
            ): m
            for m in pending
        }
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                summary = fut.result()
                if summary:
                    persist({m["id"]: summary}, [m])
                else:
                    # The backend returned nothing (or something the
                    # post-processor rejected). Record it so we can
                    # distinguish "never tried" from "tried, empty".
                    persist({}, [m], {m["id"]: "empty_output"})
            except Exception as e:
                logger.warning(f"summarize failed for {m['id']}: {e!s}")
                persist({}, [m], {m["id"]: f"{type(e).__name__}: {e!s}"})
            processed += 1
            if progress and time.time() - last_log > 3:
                _log_progress(processed, total, start)
                last_log = time.time()


def _run_batched(client, pending, concurrency, batch_size, backend, persist, progress, start, total):
    """Chunk `pending` into `batch_size`-sized batches and submit each
    batch as one future. Concurrency is the number of IN-FLIGHT BATCHES,
    each of which internally does one LLM call.
    """
    last_log = start
    processed = 0
    batches = [
        [
            {
                "id": m["id"],
                "from_addr": m["from_addr"],
                "subject": m["subject"],
                "body_text": m["body_text"],
                "labels_json": m["labels"],
            }
            for m in pending[i : i + batch_size]
        ]
        for i in range(0, len(pending), batch_size)
    ]
    # `pending` rows carry "labels"; our persist callback expects the
    # original shape, so keep a parallel list for it.
    originals = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(summarize_batch, client, batch, backend): i for i, batch in enumerate(batches)}
        for fut in as_completed(futures):
            i = futures[fut]
            errors: dict[str, str] = {}
            try:
                summaries = fut.result()
            except Exception as e:
                logger.warning(f"batch failed: {e!s}")
                summaries = {}
                err = f"batch {type(e).__name__}: {e!s}"
                errors = {m["id"]: err for m in originals[i]}
            persist(summaries, originals[i], errors)
            processed += len(originals[i])
            if progress and time.time() - last_log > 3:
                _log_progress(processed, total, start)
                last_log = time.time()


def _log_progress(processed: int, total: int, start: float) -> None:
    elapsed = time.time() - start
    rate = processed / max(elapsed, 0.001)
    eta = (total - processed) / max(rate, 0.001)
    logger.info(f"summarize: {processed}/{total} ({rate:.2f}/s, eta {eta / 60:.1f}min)")


# Read-path philosophy: writes are versioned by (model+prompt_version)
# so bumping PROMPT_VERSION correctly re-queues everything for
# re-summarization — but the UI just wants the *freshest* summary
# we have for a given message, regardless of which version produced
# it. If we keyed reads to the current DEFAULT_MODEL, the running
# server would silently return nothing after any version bump (until
# restart AND until re-summarization caught up). So reads pick the
# most recent row per message_id, full stop.


def get_summary(conn, message_id: str, model: str | None = None) -> str | None:
    """Return the most recent summary for a message, regardless of
    which model/prompt produced it. Pass `model` only when you need
    a specific key (e.g. the backfill worker checking done-ness).
    """
    if model is not None:
        row = conn.execute(
            "SELECT summary FROM message_summaries WHERE message_id = %s AND model = %s",
            (message_id, model),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT summary FROM message_summaries WHERE message_id = %s ORDER BY created_at DESC LIMIT 1",
            (message_id,),
        ).fetchone()
    return row["summary"] if row else None


def get_summaries_bulk(conn, message_ids: Iterable[str], model: str | None = None) -> dict[str, str]:
    """Like `get_summary` but bulk. When `model is None`, returns the
    freshest summary per message regardless of version — the default
    for UI reads.
    """
    rows = get_summaries_bulk_meta(conn, message_ids, model=model)
    return {mid: meta["summary"] for mid, meta in rows.items()}


def get_summaries_bulk_meta(conn, message_ids: Iterable[str], model: str | None = None) -> dict[str, dict]:
    """Same lookup as `get_summaries_bulk` but returns
    `{summary, model, created_at}` per message — useful for the
    search UI's debug panel so you can tell at a glance which prompt
    version produced a given row, or whether an old qwen-era summary
    is still being surfaced pre-re-summarize.
    """
    ids = list(message_ids)
    if not ids:
        return {}
    placeholders = ",".join(["%s"] * len(ids))
    if model is not None:
        rows = conn.execute(
            f"""SELECT message_id, summary, model, created_at
                FROM message_summaries
                WHERE message_id IN ({placeholders}) AND model = %s""",
            [*ids, model],
        ).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT message_id, summary, model, created_at
                FROM message_summaries ms
                WHERE message_id IN ({placeholders})
                  AND created_at = (
                      SELECT MAX(created_at) FROM message_summaries
                      WHERE message_id = ms.message_id
                  )""",
            ids,
        ).fetchall()
    return {
        r["message_id"]: {
            "summary": r["summary"],
            "model": r["model"],
            "created_at": r["created_at"],
        }
        for r in rows
    }
