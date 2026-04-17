"""Per-message summarization via a local Ollama model.

The summary is stored in `message_summaries` and surfaced in search
results so the agent doesn't need to fetch full thread bodies for the
common case.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import httpx

from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b"
MAX_BODY_CHARS = 8000  # ~2k tokens; keeps prompt + output under context
HTTP_TIMEOUT = 60.0

_SYSTEM_PROMPT = (
    "You summarize emails for a retrieval index. Output 1-3 sentences. "
    "Capture who sent it, key facts (amounts, dates, names, decisions), "
    "and the ask or next step. Be precise and terse — under 400 "
    "characters. Do NOT start with 'This email...' or 'The email...'. "
    "No quotation marks around the summary. No preamble."
)


def _build_user_prompt(from_addr: str, subject: str, body_text: str) -> str:
    body = (body_text or "").strip()[:MAX_BODY_CHARS]
    return f"From: {from_addr}\nSubject: {subject}\n\n{body}"


def summarize_one(
    client: httpx.Client,
    *,
    from_addr: str,
    subject: str,
    body_text: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Call Ollama and return the cleaned summary string."""
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
                "num_predict": 160,  # ~400 chars hard cap
            },
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = (data.get("message") or {}).get("content", "").strip()
    # Drop surrounding quotes and "Summary:" prefixes the model occasionally adds.
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()
    for prefix in ("Summary:", "SUMMARY:", "Subject:"):
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix) :].strip()
    return raw


def _messages_needing_summary(conn: sqlite3.Connection, model: str, limit: int | None) -> list[dict]:
    sql = """
        SELECT m.id, m.from_addr, m.subject, m.body_text
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
        {"id": r["id"], "from_addr": r["from_addr"], "subject": r["subject"], "body_text": r["body_text"]} for r in rows
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
    concurrency: int = 4,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    """Summarize every message that doesn't yet have a summary for `model`.

    Commits one message at a time so a crash mid-run doesn't lose work.
    Returns counters.
    """
    conn = get_connection(db_path)
    pending = _messages_needing_summary(conn, model, limit)
    total = len(pending)
    if total == 0:
        conn.close()
        return {"total": 0, "done": 0, "failed": 0, "seconds": 0.0}

    done = 0
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
                        f"ok={done} fail={failed})"
                    )
                    last_log = time.time()

    conn.close()
    elapsed = time.time() - start
    return {"total": total, "done": done, "failed": failed, "seconds": round(elapsed, 1)}


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
        f"SELECT message_id, summary FROM message_summaries " f"WHERE message_id IN ({placeholders}) AND model = ?",
        [*ids, model],
    ).fetchall()
    return {r["message_id"]: r["summary"] for r in rows}
