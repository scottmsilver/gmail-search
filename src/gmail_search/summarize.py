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

from gmail_search.llm import get_backend
from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)

# The `model_id` recorded in message_summaries.model is provided by
# whichever backend get_backend() returns, so switching backends (or
# models within a backend) naturally triggers a re-summarize pass —
# old rows with a different model_id aren't counted as "done".
DEFAULT_MODEL = get_backend().model_id

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
) -> str:
    """Summarize one message. Auto-classify short-circuit first, LLM fallback."""
    auto = _auto_mail_summary(labels_json, from_addr)
    if auto is not None:
        return auto

    raw = backend.chat(
        client,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(from_addr, subject, body_text)},
        ],
        max_tokens=160,
    )
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
    # jobs. model_id is the key under which summaries are stored; a
    # backend/model change naturally re-queues everything.
    backend = get_backend()
    model = backend.model_id

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

    def _persist(summaries_by_id: dict[str, str], batch_messages: list[dict]) -> None:
        nonlocal done, auto_classified, failed
        for m in batch_messages:
            summary = summaries_by_id.get(m["id"])
            if summary:
                _store_summary(conn, m["id"], summary, model)
                done += 1
                if _auto_mail_summary(m["labels"], m["from_addr"]) is not None:
                    auto_classified += 1
            else:
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
            ): m
            for m in pending
        }
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                summary = fut.result()
                persist({m["id"]: summary} if summary else {}, [m])
            except Exception as e:
                logger.warning(f"summarize failed for {m['id']}: {e!s}")
                persist({}, [m])
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
            try:
                summaries = fut.result()
            except Exception as e:
                logger.warning(f"batch failed: {e!s}")
                summaries = {}
            persist(summaries, originals[i])
            processed += len(originals[i])
            if progress and time.time() - last_log > 3:
                _log_progress(processed, total, start)
                last_log = time.time()


def _log_progress(processed: int, total: int, start: float) -> None:
    elapsed = time.time() - start
    rate = processed / max(elapsed, 0.001)
    eta = (total - processed) / max(rate, 0.001)
    logger.info(f"summarize: {processed}/{total} ({rate:.2f}/s, eta {eta / 60:.1f}min)")


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
