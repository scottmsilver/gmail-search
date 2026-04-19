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
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import httpx

from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:latest"
MAX_BODY_CHARS = 6000  # head cap before head+tail truncation
TAIL_CHARS = 1500  # for long bodies, keep this many from the end
HTTP_TIMEOUT = 120.0

_SYSTEM_PROMPT = """You summarize emails for a retrieval index.

Output 1-2 sentences, under 300 characters total.

Capture: who sent it, specific facts (amounts, dates, names, decisions),
and any explicit ask. If the sender does NOT ask for anything, just stop —
do not add filler like "no action required" or "no next step mentioned".

Do NOT begin with "This email...", "The email...", "It looks like...",
"Based on...", "The message...", "This appears...", or similar preamble.
Jump straight into the content.

Do NOT wrap the output in quotes or prefix it with "Summary:", "TLDR:", etc.

Examples of the style you should match:
- "Rebecca asks Rick to confirm whether the wood wall can move 12 inches right for the kitchen design."
- "OpenAI charged $5.78 to card ending 9535 for API credit."
- "Salvador confirms the crane arrives Thursday 11/7 with an $8,400 overage; needs go/no-go on the schedule shift."
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
    if "CATEGORY_FORUMS" in labels:
        return f"Mailing-list post from {sender}."

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


def _build_user_prompt(from_addr: str, subject: str, body_text: str) -> str:
    body = _truncate_body(_clean_body(body_text or ""))
    return f"From: {from_addr}\nSubject: {subject}\n\n{body}"


# ─── LLM call ─────────────────────────────────────────────────────────────


_STRIP_PREFIXES = ("Summary:", "SUMMARY:", "Subject:", "TLDR:", "TL;DR:")


def _clean_llm_output(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()
    for prefix in _STRIP_PREFIXES:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix) :].strip()
    return raw


def summarize_one(
    client: httpx.Client,
    *,
    from_addr: str,
    subject: str,
    body_text: str,
    labels_json: str = "[]",
    model: str = DEFAULT_MODEL,
) -> str:
    """Summarize one message. Auto-classify short-circuit first, LLM fallback."""
    auto = _auto_mail_summary(labels_json, from_addr)
    if auto is not None:
        return auto

    resp = client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(from_addr, subject, body_text)},
            ],
            "options": {
                "temperature": 0.1,
                "num_predict": 160,
            },
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = (data.get("message") or {}).get("content", "")
    return _clean_llm_output(raw)


def _messages_needing_summary(conn: sqlite3.Connection, model: str, limit: int | None) -> list[dict]:
    sql = """
        SELECT m.id, m.from_addr, m.subject, m.body_text, m.labels
        FROM messages m
        LEFT JOIN message_summaries s
          ON s.message_id = m.id AND s.model = ?
        WHERE s.message_id IS NULL
          AND length(m.body_text) > 20
        ORDER BY m.date DESC
    """
    params: list = [model]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r["id"],
            "from_addr": r["from_addr"],
            "subject": r["subject"],
            "body_text": r["body_text"],
            "labels": r["labels"],
        }
        for r in rows
    ]


def _store_summary(conn: sqlite3.Connection, message_id: str, summary: str, model: str) -> None:
    conn.execute(
        """INSERT INTO message_summaries (message_id, summary, model)
           VALUES (?, ?, ?)
           ON CONFLICT(message_id) DO UPDATE SET
             summary = excluded.summary,
             model = excluded.model,
             created_at = CURRENT_TIMESTAMP""",
        (message_id, summary, model),
    )


def backfill(
    db_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 12,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    """Summarize every message that doesn't yet have a summary for `model`.

    Commits one message at a time so a crash mid-run doesn't lose work.
    Returns counters, including `auto_classified` for mail short-circuited
    by `_auto_mail_summary` without an LLM call.
    """
    conn = get_connection(db_path)
    pending = _messages_needing_summary(conn, model, limit)
    total = len(pending)
    if total == 0:
        conn.close()
        return {"total": 0, "done": 0, "auto_classified": 0, "failed": 0, "seconds": 0.0}

    done = 0
    auto_classified = 0
    failed = 0
    start = time.time()
    last_log = start

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    summarize_one,
                    client,
                    from_addr=m["from_addr"],
                    subject=m["subject"],
                    body_text=m["body_text"],
                    labels_json=m["labels"],
                    model=model,
                ): m
                for m in pending
            }
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    summary = fut.result()
                    if summary:
                        _store_summary(conn, m["id"], summary, model)
                        conn.commit()
                        done += 1
                        if _auto_mail_summary(m["labels"], m["from_addr"]) is not None:
                            auto_classified += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.warning(f"summarize failed for {m['id']}: {e!s}")
                if progress and time.time() - last_log > 3:
                    elapsed = time.time() - start
                    rate = (done + failed) / max(elapsed, 0.001)
                    eta = (total - done - failed) / max(rate, 0.001)
                    logger.info(
                        f"summarize: {done + failed}/{total} "
                        f"({rate:.1f}/s, eta {eta / 60:.1f}min, "
                        f"ok={done} auto={auto_classified} fail={failed})"
                    )
                    last_log = time.time()

    conn.close()
    elapsed = time.time() - start
    return {
        "total": total,
        "done": done,
        "auto_classified": auto_classified,
        "failed": failed,
        "seconds": round(elapsed, 1),
    }


def get_summary(conn: sqlite3.Connection, message_id: str, model: str = DEFAULT_MODEL) -> str | None:
    row = conn.execute(
        "SELECT summary FROM message_summaries WHERE message_id = ? AND model = ?",
        (message_id, model),
    ).fetchone()
    return row["summary"] if row else None


def get_summaries_bulk(
    conn: sqlite3.Connection, message_ids: Iterable[str], model: str = DEFAULT_MODEL
) -> dict[str, str]:
    ids = list(message_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT message_id, summary FROM message_summaries WHERE message_id IN ({placeholders}) AND model = ?",
        [*ids, model],
    ).fetchall()
    return {r["message_id"]: r["summary"] for r in rows}
