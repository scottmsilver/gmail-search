import json
from datetime import datetime
from typing import Optional

from gmail_search.auth.write_user import resolve_write_user_id
from gmail_search.store.models import Attachment, EmbeddingRecord, Message


def upsert_message(conn, msg: Message, *, user_id: Optional[str] = None) -> None:
    uid = resolve_write_user_id(conn, user_id=user_id)
    conn.execute(
        """INSERT INTO messages (id, thread_id, from_addr, to_addr, subject,
           body_text, body_html, date, labels, history_id, raw_json, user_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT(id) DO UPDATE SET
             thread_id=excluded.thread_id, from_addr=excluded.from_addr,
             to_addr=excluded.to_addr, subject=excluded.subject,
             body_text=excluded.body_text, body_html=excluded.body_html,
             date=excluded.date, labels=excluded.labels,
             history_id=excluded.history_id, raw_json=excluded.raw_json""",
        (
            msg.id,
            msg.thread_id,
            msg.from_addr,
            msg.to_addr,
            msg.subject,
            msg.body_text,
            msg.body_html,
            msg.date.isoformat(),
            json.dumps(msg.labels),
            msg.history_id,
            msg.raw_json,
            uid,
        ),
    )
    _touch_thread_summary(conn, msg, user_id=uid)
    conn.commit()


def _touch_thread_summary(conn, msg: Message, *, user_id: Optional[str] = None) -> None:
    """Fast-path inline refresh fired from `upsert_message`. Wraps
    `recompute_thread_summary` so the reconciler and the ingest hot
    path share one implementation.
    """
    recompute_thread_summary(conn, msg.thread_id, fallback_subject=msg.subject, user_id=user_id)


def recompute_thread_summary(
    conn, thread_id: str, fallback_subject: str = "", *, user_id: Optional[str] = None
) -> bool:
    """Refresh the thread_summary row for one thread from the current
    `messages` content. Returns True on a write, False when the
    thread has no messages (deleted / race).

    Called from two places:
      * `_touch_thread_summary` — inline during ingest, low-latency
        happy path.
      * `gmail-search reconcile` — background drift-detector daemon
        that sweeps any rows missed by the inline path. That's what
        makes the inline call a fast path rather than a correctness
        requirement.
    """
    # Scope the aggregation to the thread's owner. Tenant-safety: without the
    # user_id filter a (vanishingly unlikely) cross-account thread_id collision
    # would mix two users' messages into one summary, and the global `reconcile`
    # daemon would otherwise resolve uid to the bootstrap user. Resolve uid first
    # and filter by it so the summary is always built from exactly one tenant's
    # mail. (No-op on current data: no thread_id spans >1 user.)
    uid = resolve_write_user_id(conn, user_id=user_id)
    row = conn.execute(
        """SELECT
             count(*)                                   AS message_count,
             min(date)                                  AS date_first,
             max(date)                                  AS date_last,
             min(subject) FILTER (WHERE date =
                 (SELECT min(date) FROM messages
                  WHERE thread_id = %s AND user_id = %s))   AS subject,
             array_agg(DISTINCT from_addr ORDER BY from_addr) AS from_addrs,
             string_agg(DISTINCT labels, '|')           AS labels_raw
           FROM messages WHERE thread_id = %s AND user_id = %s""",
        (thread_id, uid, thread_id, uid),
    ).fetchone()
    if row is None or not row["message_count"]:
        return False
    labels: set[str] = set()
    for chunk in (row["labels_raw"] or "").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            labels.update(json.loads(chunk))
        except Exception:
            continue
    from_addrs = list(row["from_addrs"] or [])
    conn.execute(
        """INSERT INTO thread_summary
             (thread_id, subject, participants, all_from_addrs, all_labels,
              message_count, date_first, date_last, user_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT(thread_id, user_id) DO UPDATE SET
             subject = excluded.subject,
             participants = excluded.participants,
             all_from_addrs = excluded.all_from_addrs,
             all_labels = excluded.all_labels,
             message_count = excluded.message_count,
             date_first = excluded.date_first,
             date_last = excluded.date_last""",
        (
            thread_id,
            row["subject"] or fallback_subject,
            json.dumps(from_addrs),
            json.dumps(from_addrs),
            json.dumps(sorted(labels)),
            row["message_count"],
            row["date_first"],
            row["date_last"],
            uid,
        ),
    )
    return True


def get_message(conn, message_id: str, *, user_id: Optional[str] = None) -> Message | None:
    """Fetch a message by id. When `user_id` is given, the lookup is
    scoped to that user — returns None if the message exists but
    belongs to a different user (treats it as not-found rather than
    leaking existence)."""
    if user_id is None:
        row = conn.execute("SELECT * FROM messages WHERE id = %s", (message_id,)).fetchone()
    else:
        row = conn.execute("SELECT * FROM messages WHERE id = %s AND user_id = %s", (message_id, user_id)).fetchone()
    if row is None:
        return None
    return Message(
        id=row["id"],
        thread_id=row["thread_id"],
        from_addr=row["from_addr"],
        to_addr=row["to_addr"],
        subject=row["subject"],
        body_text=row["body_text"],
        body_html=row["body_html"],
        date=datetime.fromisoformat(row["date"]),
        labels=json.loads(row["labels"]),
        history_id=row["history_id"],
        raw_json=row["raw_json"],
    )


def get_message_bodies_bulk(conn, message_ids: list[str], *, user_id: Optional[str] = None) -> dict[str, str]:
    """Fetch plain-text bodies for many messages at once, keyed by id.

    Used by the search endpoint's `match_detail=full` mode so an agent can
    read whole emails per match in one round-trip instead of N get_thread
    calls. Scoped to `user_id` when given so one user's search can never
    surface another user's body text."""
    if not message_ids:
        return {}
    ids = list(message_ids)
    if user_id is not None:
        rows = conn.execute(
            "SELECT id, body_text FROM messages WHERE id = ANY(%s) AND user_id = %s",
            (ids, user_id),
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, body_text FROM messages WHERE id = ANY(%s)", (ids,)).fetchall()
    return {r["id"]: (r["body_text"] or "") for r in rows}


def get_messages_without_embeddings(
    conn, model: str, *, user_id: Optional[str] = None, limit: Optional[int] = None
) -> list[Message]:
    """Messages that have no embedding row for `model` yet. When
    `user_id` is given, scopes to that user — required for per-user
    daemons so silvershabbat's embed pass doesn't try to embed
    scott's 410k messages too (and OOM the box).

    `limit` bounds the batch: the old unbounded `SELECT m.* … fetchall()`
    loaded EVERY unembedded message's body + raw_json into RAM at once
    (~18 GB for a 220k backlog) and OOM-killed the embed step. Callers
    embed in chunks and loop. Uses NOT EXISTS + ORDER BY id so the LIMIT
    can terminate early via the PK + embeddings indexes.
    """
    lim_sql = " ORDER BY m.id LIMIT %s" if limit is not None else ""
    if user_id is not None:
        sql = (
            "SELECT m.* FROM messages m WHERE m.user_id = %s "
            "AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.message_id = m.id "
            "AND e.chunk_type = 'message' AND e.model = %s AND e.user_id = %s)" + lim_sql
        )
        params = (user_id, model, user_id) + ((limit,) if limit is not None else ())
    else:
        sql = (
            "SELECT m.* FROM messages m WHERE NOT EXISTS (SELECT 1 FROM embeddings e "
            "WHERE e.message_id = m.id AND e.chunk_type = 'message' AND e.model = %s)" + lim_sql
        )
        params = (model,) + ((limit,) if limit is not None else ())
    rows = conn.execute(sql, params).fetchall()
    return [
        Message(
            id=r["id"],
            thread_id=r["thread_id"],
            from_addr=r["from_addr"],
            to_addr=r["to_addr"],
            subject=r["subject"],
            body_text=r["body_text"],
            body_html=r["body_html"],
            date=datetime.fromisoformat(r["date"]),
            labels=json.loads(r["labels"]),
            history_id=r["history_id"],
            raw_json=r["raw_json"],
        )
        for r in rows
    ]


def upsert_attachment(conn, att: Attachment, *, user_id: Optional[str] = None) -> int:
    # `RETURNING id` — portable across SQLite 3.35+ and Postgres, and the
    # only reliable way to get the newly-generated BIGSERIAL from psycopg.
    uid = resolve_write_user_id(conn, user_id=user_id)
    cursor = conn.execute(
        """INSERT INTO attachments (message_id, filename, mime_type, size_bytes,
           extracted_text, image_path, raw_path, user_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT(message_id, filename) DO UPDATE SET
             mime_type=excluded.mime_type, size_bytes=excluded.size_bytes,
             raw_path=excluded.raw_path
           RETURNING id""",
        (
            att.message_id,
            att.filename,
            att.mime_type,
            att.size_bytes,
            att.extracted_text,
            att.image_path,
            att.raw_path,
            uid,
        ),
    )
    row = cursor.fetchone()
    conn.commit()
    if row is None:
        return 0
    try:
        return int(row["id"])
    except (KeyError, TypeError):
        return int(row[0])


def get_attachments_for_message(conn, message_id: str, *, user_id: Optional[str] = None) -> list[Attachment]:
    if user_id is None:
        rows = conn.execute("SELECT * FROM attachments WHERE message_id = %s", (message_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM attachments WHERE message_id = %s AND user_id = %s",
            (message_id, user_id),
        ).fetchall()
    return [
        Attachment(
            id=r["id"],
            message_id=r["message_id"],
            filename=r["filename"],
            mime_type=r["mime_type"],
            size_bytes=r["size_bytes"],
            extracted_text=r["extracted_text"],
            image_path=r["image_path"],
            raw_path=r["raw_path"],
        )
        for r in rows
    ]


def get_pending_extraction_message_ids(conn, *, user_id: Optional[str] = None) -> list[str]:
    """Return distinct message ids that have at least one local-file
    attachment still awaiting extraction (`extracted_text IS NULL`,
    `image_path IS NULL`, `raw_path IS NOT NULL`), ordered so that
    frontfill wins over backfill — same ordering idiom the summarizer
    uses (see `summarize._fetch_messages_needing_summary`): messages
    received in the last 24h go first, then everything else by date
    DESC.

    This prevents the `update --loop` daemon from perpetually grinding
    through the multi-year backfill while fresh arrivals from the
    `watch` daemon sit unextracted. Callers iterate the list and pass
    each id through `get_attachments_for_message` to dispatch the
    actual extractor.

    Multi-tenant: when `user_id` is given, only that user's pending
    attachments are returned. WITHOUT this filter a per-user daemon
    silently extracts every other user's backlog too — silvershabbat's
    daemon was OOM-killed after pulling 14k of scott's pending
    attachments into memory before getting to embed its own 118.
    """
    # GROUP BY (not DISTINCT) so we can reference MAX(m.date) in ORDER
    # BY — Postgres forbids `SELECT DISTINCT … ORDER BY <expr not in
    # select list>`. A given m.id has exactly one m.date, so MAX is a
    # no-op aggregation, just a syntactic bridge.
    user_clause = "AND m.user_id = %s" if user_id is not None else ""
    params = (user_id,) if user_id is not None else ()
    rows = conn.execute(
        f"""
        SELECT m.id
        FROM messages m
        JOIN attachments a ON a.message_id = m.id
        WHERE a.extracted_text IS NULL
          AND a.image_path IS NULL
          AND a.raw_path IS NOT NULL
          {user_clause}
        GROUP BY m.id
        ORDER BY
          -- Frontfill wins over backfill: anything received in the
          -- last 24h (i.e. what `watch` / `sync_new_messages` just
          -- pulled) goes to the top of the queue, ahead of any
          -- older unextracted backlog.
          (MAX(m.date::timestamptz) > NOW() - INTERVAL '1 day') DESC,
          MAX(m.date) DESC
        """,
        params,
    ).fetchall()
    return [r["id"] for r in rows]


def extract_attachment_on_demand(conn, attachment_id: int, *, config: dict):
    """Run the extractor for a single attachment that has NULL/empty
    `extracted_text`, write the result back to the row, and return the
    fresh `ExtractResult`.

    Return semantics:
      * `None` when there is nothing to do / nothing we can do — the row
        doesn't exist, extracted_text is already populated, raw_path is
        missing from disk, or no extractor matches the mime type.
      * `ExtractResult` on success (text and/or image paths).

    Fallback for the `/api/attachment/<id>/text` endpoint so freshly
    arrived attachments don't have to wait for the `update --loop`
    daemon's extract pass. The happy path for anything already
    extracted stays zero-cost — the early return on non-empty
    `extracted_text` means we never re-run the extractor for rows the
    daemon has already processed.
    """
    from pathlib import Path as _Path

    from gmail_search.extract import dispatch

    row = conn.execute(
        "SELECT id, mime_type, extracted_text, image_path, raw_path " "FROM attachments WHERE id = %s",
        (attachment_id,),
    ).fetchone()
    if row is None:
        return None
    if (row["extracted_text"] or "").strip():
        return None
    if not row["raw_path"]:
        return None
    raw_path = _Path(row["raw_path"])
    if not raw_path.exists():
        return None

    att_config = (config or {}).get("attachments", {})
    result = dispatch(row["mime_type"], raw_path, att_config)
    if result is None:
        return None

    updates: dict = {}
    if result.text:
        updates["extracted_text"] = result.text
    if result.images:
        first = result.images[0]
        updates["image_path"] = str(first.parent if len(result.images) > 1 else first)
    if not updates:
        return result

    set_clause = ", ".join(f"{k} = %s" for k in updates)
    conn.execute(
        f"UPDATE attachments SET {set_clause} WHERE id = %s",
        (*updates.values(), attachment_id),
    )
    conn.commit()
    return result


# ─── Drive stubs ───────────────────────────────────────────────────────────
# Drive-linked docs don't come from Gmail attachments; they're referenced
# by URL from the message body. We represent them as attachment rows with
# `mime_type = application/vnd.google-apps.*`, no `raw_path` (content is
# remote), and a `filename` that encodes the drive_id so the fetch step
# can round-trip to it: "Drive: [<drive_id>]" as a stub, then
# "Drive: <title> [<drive_id>]" once fetched.


def upsert_drive_stub(
    conn,
    *,
    message_id: str,
    drive_id: str,
    mime_type: str,
    user_id: Optional[str] = None,
) -> int:
    """Insert a stub row for a Drive-linked doc. Returns the number
    of new rows inserted (0 if the stub already existed; the
    `(message_id, filename)` dedup index enforces idempotency).
    """
    filename = f"Drive: [{drive_id}]"
    uid = resolve_write_user_id(conn, user_id=user_id)
    cursor = conn.execute(
        """INSERT INTO attachments
           (message_id, filename, mime_type, size_bytes, user_id)
           VALUES (%s, %s, %s, 0, %s)
           ON CONFLICT (message_id, filename) DO NOTHING""",
        (message_id, filename, mime_type, uid),
    )
    return cursor.rowcount


def set_active_index_dir(conn, path: str, *, user_id: Optional[str] = None) -> None:
    """Flip the per-user `scann_index_pointer` row to a new on-disk
    path. Readers resolve the active index through this row so a
    reindex swap is atomic at the DB layer — no reliance on filesystem
    rename semantics. See `build_index_sharded` for the writer side.
    """
    uid = resolve_write_user_id(conn, user_id=user_id)
    conn.execute(
        """INSERT INTO scann_index_pointer (user_id, current_dir, updated_at)
           VALUES (%s, %s, CURRENT_TIMESTAMP)
           ON CONFLICT(user_id) DO UPDATE SET
             current_dir = excluded.current_dir,
             updated_at = CURRENT_TIMESTAMP""",
        (uid, path),
    )


def get_active_index_dir(conn, *, user_id: Optional[str] = None) -> str | None:
    uid = resolve_write_user_id(conn, user_id=user_id)
    row = conn.execute("SELECT current_dir FROM scann_index_pointer WHERE user_id = %s", (uid,)).fetchone()
    return row["current_dir"] if row else None


def fill_drive_attachment(
    conn,
    *,
    attachment_id: int,
    title: str,
    text: str,
    drive_id: str,
) -> None:
    """Populate a previously-stubbed Drive row with fetched content.
    Renames the filename to include the title so the UI shows it.
    """
    new_filename = f"Drive: {title} [{drive_id}]"
    conn.execute(
        "UPDATE attachments SET extracted_text = %s, filename = %s, size_bytes = %s WHERE id = %s",
        (text, new_filename, len(text), attachment_id),
    )


# ─── URL stubs ─────────────────────────────────────────────────────────────
# Mirrors the Drive-stub pattern for arbitrary URLs linked from an email
# body. A URL stub is an `attachments` row with `mime_type = 'text/html'`,
# `filename = "URL: <url>"` (or `"URL: <title> [<url>]"` once filled),
# and `extracted_text = NULL` until the fetcher populates it.
#
# The UNIQUE(message_id, filename) constraint is what gives us
# idempotency — inserting the same stub twice is a no-op.

# Keep stub filenames bounded so a pathological URL doesn't blow up
# downstream code that logs or renders the filename. The DB col is TEXT
# so there's no hard limit, but BM25 / UI rendering / regex scans all
# prefer something sane.
_URL_STUB_FILENAME_CAP = 500


def _url_stub_filename(url: str) -> str:
    """Canonical stub filename `"URL: <url>"`, capped for sanity."""
    base = f"URL: {url}"
    return base[:_URL_STUB_FILENAME_CAP]


def upsert_url_stub(conn, *, message_id: str, url: str, user_id: Optional[str] = None) -> int:
    """Insert a stub row for a URL linked from the message body.

    Returns the number of rows inserted (0 if the stub already
    existed; dedup via UNIQUE(message_id, filename)). Matches the
    shape of `upsert_drive_stub`.
    """
    filename = _url_stub_filename(url)
    uid = resolve_write_user_id(conn, user_id=user_id)
    cursor = conn.execute(
        """INSERT INTO attachments
           (message_id, filename, mime_type, size_bytes, user_id)
           VALUES (%s, %s, 'text/html', 0, %s)
           ON CONFLICT (message_id, filename) DO NOTHING""",
        (message_id, filename, uid),
    )
    return cursor.rowcount


def get_crawl_blocked_reason(conn, *, message_id: str) -> Optional[str]:
    """Return the cached invitation-guard verdict for a message, or None
    if the message was never gated (the common case). Lets a re-sync
    reuse the verdict instead of re-calling Gemini. A row that doesn't
    exist yet also returns None."""
    row = conn.execute(
        "SELECT crawl_blocked_reason FROM messages WHERE id = %s",
        (message_id,),
    ).fetchone()
    return row["crawl_blocked_reason"] if row else None


def set_crawl_blocked_reason(conn, *, message_id: str, reason: Optional[str]) -> None:
    """Persist the invitation-guard verdict for a message. `reason` is a
    short human string when all links were skipped, or None when the
    message crawls normally. Idempotent — overwrites any prior verdict."""
    conn.execute(
        "UPDATE messages SET crawl_blocked_reason = %s WHERE id = %s",
        (reason, message_id),
    )


def fill_url_attachment(
    conn,
    *,
    attachment_id: int,
    title: str,
    text: str,
    url: str,
) -> None:
    """Populate a SINGLE stubbed URL row with fetched content, so it gets
    embedded exactly once. The same URL is linked from up to thousands of
    messages; the page is identical, so we embed one representative copy (this
    one) and the caller abandons the duplicate stubs WITHOUT content — embedding
    the same page N times would just bloat the ScaNN index and churn reindex
    without improving recall. Renames to `"URL: <title> [<url>]"` so the UI is
    human and `url_from_stub_filename` still round-trips.
    """
    display = (title or _host_of(url) or "link").strip()
    new_filename = f"URL: {display} [{url}]"[:_URL_STUB_FILENAME_CAP]
    conn.execute(
        "UPDATE attachments SET extracted_text = %s, filename = %s, size_bytes = %s WHERE id = %s",
        (text, new_filename, len(text), attachment_id),
    )


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def url_from_stub_filename(filename: str) -> str | None:
    """Extract the URL out of a `URL: <url>` (unfilled) or
    `URL: <title> [<url>]` (filled) stub filename. Returns None if
    the filename isn't a URL stub.

    Mirrors `drive_id_from_stub_filename` but without a strict
    charset check — URLs are validated at fetch time by the SSRF
    guard in `url_fetcher`, which is the real authority.
    """
    if not filename or not filename.startswith("URL:"):
        return None
    # Filled shape: "URL: <title> [<url>]"
    if filename.endswith("]") and "[" in filename:
        candidate = filename.rsplit("[", 1)[1].rstrip("]").strip()
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate
    # Unfilled shape: "URL: <url>"
    rest = filename[len("URL:") :].strip()
    if rest.startswith("http://") or rest.startswith("https://"):
        return rest
    return None


# After this many failed crawl attempts a URL stub is abandoned (dead /
# anti-bot link). Each attempt also backs off `crawl_attempts` hours, so a
# stub is retried at ~1h, 2h, 3h then dropped — it can no longer head-of-line
# block live URLs in `pending_url_stubs`.
_MAX_CRAWL_ATTEMPTS = 4


def pending_url_stubs(conn, limit: int) -> list[dict]:
    """Return URL stubs that haven't been fetched yet, oldest message
    first. Each dict carries `{id, message_id, url}`.

    Rows whose URL is now on the denylist (the list can tighten over
    time — see `gmail/url_extract.py::_is_denied`) are deleted here so
    the crawler never wastes a Chromium render on them. This is a
    self-healing purge: any time the crawler runs, the next batch
    sweeps newly-denied pending rows off the table.

    Intentionally NOT scoped by user_id. The crawler is a SINGLE global
    daemon (`gmail-search crawl --loop`, not one per tenant): URL
    reachability is a property of the URL, not the mailbox, so one daemon
    fetches every tenant's stubs. Cross-tenant safety lives downstream —
    `url_fetcher._write_result_sync` fills one representative PER USER, so
    each tenant's index gets the page independently. The denylist is a
    global policy, so purging a denied URL for all users is the intended
    behavior, not a cross-user write. (Verified 2026-07-08; don't add
    `WHERE user_id` here — it would starve every non-bootstrap tenant.)
    """
    # Scope to our stub shape so real HTML email attachments
    # (`message.html`, `ATT00001.htm` etc — also mime='text/html')
    # don't get picked up by the crawler. Those are filled by the
    # local-extract path, not the URL crawler.
    #
    # LIKE pattern bound as a param because psycopg treats bare `%`
    # in the SQL text as a placeholder marker.
    #
    # Overfetch by 5× so that even when most of the batch is denied
    # we still return `limit` real URLs. Cheap: the filter runs in
    # Python, and denied rows get deleted in the same pass.
    from gmail_search.gmail.url_extract import _is_denied

    # Fast lane / slow lane to avoid head-of-line blocking. A failed crawl
    # leaves extracted_text NULL, so without attempt-tracking the SAME dead /
    # anti-bot links were re-selected EVERY cycle forever — the newest 200 dead
    # stubs permanently filled the batch and live URLs behind them never ran.
    # Now: never-tried stubs (crawl_attempts=0) are crawled FIRST (fast lane);
    # previously-failed ones back off (retry only after `attempts` hours) and
    # are abandoned after _MAX_CRAWL_ATTEMPTS (slow lane). `_process_one` stamps
    # each attempt. The partial index idx_attachments_crawl_lane serves the
    # (crawl_attempts ASC, id DESC) order directly.
    overfetch = max(limit * 5, 500)
    rows = conn.execute(
        """SELECT id, message_id, filename
             FROM attachments
            WHERE mime_type = 'text/html'
              AND extracted_text IS NULL
              AND filename LIKE %s
              AND crawl_attempts < %s
              AND (crawl_last_attempt IS NULL
                   OR crawl_last_attempt < now() - (interval '1 hour' * crawl_attempts))
            ORDER BY crawl_attempts ASC, id DESC
            LIMIT %s""",
        ("URL: %", _MAX_CRAWL_ATTEMPTS, overfetch),
    ).fetchall()

    out: list[dict] = []
    deny_ids: list[int] = []
    seen_urls: set[str] = set()
    for r in rows:
        url = url_from_stub_filename(r["filename"])
        if not url:
            deny_ids.append(int(r["id"]))
            continue
        if _is_denied(url):
            deny_ids.append(int(r["id"]))
            continue
        # Dedup within the batch: the same URL is linked from many messages,
        # and `fill_url_attachment` fans a single fetch out to every copy, so
        # crawling one stub resolves them all. Carrying `filename` lets the
        # fetcher fill/mark all copies by their exact original stub filename.
        if url in seen_urls:
            continue
        seen_urls.add(url)
        out.append({"id": int(r["id"]), "message_id": r["message_id"], "url": url, "filename": r["filename"]})
        if len(out) >= limit:
            break

    if deny_ids:
        B = 1000
        for i in range(0, len(deny_ids), B):
            batch = deny_ids[i : i + B]
            placeholders = ",".join(["%s"] * len(batch))
            conn.execute(f"DELETE FROM attachments WHERE id IN ({placeholders})", batch)
        conn.commit()

    return out


def insert_embedding(conn, rec: EmbeddingRecord, *, user_id: Optional[str] = None) -> int:
    # `RETURNING id` — portable across SQLite 3.35+ and Postgres, and
    # the only reliable way to get the newly-generated BIGSERIAL from
    # psycopg.
    uid = resolve_write_user_id(conn, user_id=user_id)
    cursor = conn.execute(
        """INSERT INTO embeddings (message_id, attachment_id, chunk_type,
           chunk_text, embedding, model, user_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           RETURNING id""",
        (rec.message_id, rec.attachment_id, rec.chunk_type, rec.chunk_text, rec.embedding, rec.model, uid),
    )
    row = cursor.fetchone()
    conn.commit()
    if row is None:
        return 0
    try:
        return int(row["id"])
    except (KeyError, TypeError):
        return int(row[0])


def embedding_exists(conn, message_id: str, attachment_id: int | None, chunk_type: str, model: str) -> bool:
    if attachment_id is None:
        row = conn.execute(
            """SELECT 1 FROM embeddings
               WHERE message_id = %s AND attachment_id IS NULL
               AND chunk_type = %s AND model = %s""",
            (message_id, chunk_type, model),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT 1 FROM embeddings
               WHERE message_id = %s AND attachment_id = %s
               AND chunk_type = %s AND model = %s""",
            (message_id, attachment_id, chunk_type, model),
        ).fetchone()
    return row is not None


def load_all_embeddings(conn, model: str) -> tuple[list[int], list[bytes]]:
    rows = conn.execute("SELECT id, embedding FROM embeddings WHERE model = %s", (model,)).fetchall()
    ids = [r["id"] for r in rows]
    blobs = [r["embedding"] for r in rows]
    return ids, blobs


def set_sync_state(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO sync_state (key, value) VALUES (%s, %s) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_sync_state(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = %s", (key,)).fetchone()
    return row["value"] if row else None


# Hardened FTS input pipeline. The model can hand us anything — quoted
# phrases, parenthesized expressions, column filters, control chars,
# enormous strings. We allow-list a tiny character set and cap sizes.
import re as _re

# Keep word chars (letters/digits/underscore), apostrophe (don't), hyphen
# (well-known), and whitespace. Everything else is collapsed to a space.
_FTS_KEEP = _re.compile(r"[^\w'\-\s]+", _re.UNICODE)
_MAX_TOKEN_LEN = 64
_MAX_TOKENS = 32
_MAX_QUERY_LEN = 2000


def _sanitize_fts_tokens(raw: str) -> list[str]:
    """Convert arbitrary user/model input into a safe list of FTS tokens.

    - Allow-lists a tiny character set (word chars, ', -, whitespace).
    - Drops empty tokens, leading/trailing punctuation, and FTS5 operator
      keywords (AND, OR, NOT, NEAR) so they cannot be injected.
    - Caps both per-token length and total token count.
    """
    if not raw:
        return []
    truncated = raw[:_MAX_QUERY_LEN]
    cleaned = _FTS_KEEP.sub(" ", truncated)
    tokens: list[str] = []
    for tok in cleaned.split():
        # Trim leading/trailing hyphens and apostrophes (they aren't useful
        # alone and could confuse FTS tokenization).
        stripped = tok.strip("-'")
        if not stripped:
            continue
        if stripped.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        tokens.append(stripped[:_MAX_TOKEN_LEN])
        if len(tokens) >= _MAX_TOKENS:
            break
    return tokens


def search_fts(
    conn,
    query: str,
    limit: int = 200,
    candidate_ids: list[str] | None = None,
    *,
    user_id: Optional[str] = None,
) -> dict[str, float]:
    """Search messages + attachments via pg_search BM25, return {message_id: score}.

    Two-pass strategy mirroring the historic SQLite FTS5 behaviour:
      1. Phrase pass (higher weight).
      2. Disjunction pass (broader recall).
    Scores are min-max normalized to [0, 1] so downstream blenders can mix
    FTS with vector similarity.

    ``candidate_ids`` optionally restricts BM25 to a pre-filtered set of
    message IDs (see SearchEngine._resolve_candidate_msg_ids). ``None``
    means no restriction within the user's corpus. ``[]`` means "filters
    matched zero rows" — short-circuit to an empty result.

    ``user_id`` scopes the BM25 to a single user's corpus. The 0d
    benchmark confirmed ParadeDB BM25 composes cleanly with
    ``AND user_id = $1`` — no recall hit, ~75ms latency.
    """
    if candidate_ids is not None and len(candidate_ids) == 0:
        return {}
    return _search_fts_postgres(conn, query, limit, candidate_ids, user_id=user_id)


def _pg_bm25_messages(
    conn,
    bm25_query: str,
    limit: int,
    logger,
    candidate_ids: list[str] | None = None,
    *,
    user_id: Optional[str] = None,
) -> dict[str, float]:
    """BM25 pass against the `messages` table via pg_search.

    Uses the `@@@` operator (Tantivy BM25 match) and `paradedb.score(id)`
    for a real BM25 score. The query string is interpreted by Tantivy's
    query parser, which natively handles phrase quoting (`"foo bar"`),
    boolean operators, and per-field targeting.

    When ``candidate_ids`` is non-None, restricts the match set to those
    message IDs via ``id = ANY(%s::text[])``. ``None`` means no
    restriction. An empty list is treated as a no-op short-circuit by the
    caller (``search_fts``).
    """
    scores: dict[str, float] = {}
    try:
        # Per the 0d benchmark, ParadeDB BM25 (`@@@`) composes cleanly
        # with `AND user_id = $1` — same recall, ~75ms latency. We add
        # the user filter into every BM25 SQL even when candidate_ids
        # is set, since candidate_ids may have come from a different
        # path that didn't itself filter by user.
        user_clause = " AND user_id = %s" if user_id else ""
        user_param: list = [user_id] if user_id else []
        if candidate_ids is None:
            rows = conn.execute(
                "SELECT id AS message_id, paradedb.score(id) AS rank "
                "FROM messages "
                f"WHERE messages @@@ %s{user_clause} "
                "ORDER BY rank DESC LIMIT %s",
                [bm25_query] + user_param + [limit],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id AS message_id, paradedb.score(id) AS rank "
                "FROM messages "
                f"WHERE messages @@@ %s AND id = ANY(%s::text[]){user_clause} "
                "ORDER BY rank DESC LIMIT %s",
                [bm25_query, list(candidate_ids)] + user_param + [limit],
            ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = float(r["rank"] or 0.0)
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except Exception as e:  # paranoid catch — FTS must never fail the request
        logger.exception(f"PG BM25 error on messages: {e!s} | query={bm25_query!r}")
    return scores


def _pg_bm25_attachments(
    conn,
    bm25_query: str,
    limit: int,
    logger,
    candidate_ids: list[str] | None = None,
    *,
    user_id: Optional[str] = None,
) -> dict[str, float]:
    """BM25 pass against `attachments`. Returns `{message_id: score}` by
    joining the attachment row's `message_id`. `paradedb.score(id)` keys
    on the attachment PK (the BM25 `key_field`), so the score is per
    attachment; multiple attachment hits on the same message collapse to
    the max score in the caller.

    When ``candidate_ids`` is non-None, restricts hits to attachments
    whose ``message_id`` is in that set.
    """
    scores: dict[str, float] = {}
    try:
        user_clause = " AND user_id = %s" if user_id else ""
        user_param: list = [user_id] if user_id else []
        if candidate_ids is None:
            rows = conn.execute(
                "SELECT message_id, paradedb.score(id) AS rank "
                "FROM attachments "
                f"WHERE attachments @@@ %s{user_clause} "
                "ORDER BY rank DESC LIMIT %s",
                [bm25_query] + user_param + [limit],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT message_id, paradedb.score(id) AS rank "
                "FROM attachments "
                f"WHERE attachments @@@ %s AND message_id = ANY(%s::text[]){user_clause} "
                "ORDER BY rank DESC LIMIT %s",
                [bm25_query, list(candidate_ids)] + user_param + [limit],
            ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = float(r["rank"] or 0.0)
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except Exception as e:
        logger.exception(f"PG BM25 error on attachments: {e!s} | query={bm25_query!r}")
    return scores


# Field lists for the BM25 multi-field query builder. pg_search's Tantivy
# parser requires explicit `field:term` syntax — an unqualified `invoice`
# searches nothing. We mirror the SQLite FTS column set: the `messages`
# FTS5 virtual table indexed {subject, from, to, body}; `attachments`
# indexed {filename, extracted_text}. Keep these in sync with the
# `USING bm25 (...)` column list in pg_schema.sql.
_BM25_MESSAGE_FIELDS = ("subject", "from_addr", "to_addr", "body_text")
_BM25_ATTACHMENT_FIELDS = ("filename", "extracted_text")


def _build_bm25_query(tokens: list[str], fields: tuple[str, ...]) -> tuple[str, str | None]:
    """Build Tantivy query strings targeting every FTS field.

    Returns `(disjunction_query, phrase_query_or_none)`:
      * Disjunction pass: `(f1:t1 f1:t2 ... f2:t1 f2:t2 ...)` — any
        token in any field matches. Tantivy's default combinator is OR.
      * Phrase pass: `(f1:"t1 t2 ..." f2:"t1 t2 ..." ...)` when there
        are ≥2 tokens — ordered-adjacent match.

    Tokens come from `_sanitize_fts_tokens()` so they are safe to
    interpolate (no quotes, backslashes, or Tantivy operators).
    """
    disjunction_terms = [f"{f}:{t}" for f in fields for t in tokens]
    disjunction = " ".join(disjunction_terms)

    phrase: str | None = None
    if len(tokens) > 1:
        phrase_body = " ".join(tokens)
        phrase = " ".join(f'{f}:"{phrase_body}"' for f in fields)
    return disjunction, phrase


def _search_fts_postgres(
    conn,
    query: str,
    limit: int,
    candidate_ids: list[str] | None = None,
    *,
    user_id: Optional[str] = None,
) -> dict[str, float]:
    """Postgres BM25 path via pg_search (paradedb / Tantivy).

    Two passes:
      1. Phrase pass — `"t1 t2 t3"` — tokens must appear in order,
         adjacent. Results get a ×1.5 boost.
      2. Disjunction pass — `t1 t2 t3` (Tantivy default = OR) — any
         sanitized term can match.

    Scores come from `paradedb.score(id)` (real BM25). We min-max
    normalize across the result set so downstream blend code sees
    values in [0, 1].

    ``candidate_ids`` optionally narrows both passes to a pre-filtered
    set of message IDs. See ``search_fts`` for semantics.
    """
    import logging

    logger = logging.getLogger(__name__)
    tokens = _sanitize_fts_tokens(query)
    if not tokens:
        return {}

    msg_disj, msg_phrase = _build_bm25_query(tokens, _BM25_MESSAGE_FIELDS)
    att_disj, att_phrase = _build_bm25_query(tokens, _BM25_ATTACHMENT_FIELDS)
    scores: dict[str, float] = {}

    # Strategy 1: Phrase match (words must appear in order, adjacent).
    if msg_phrase is not None:
        for tbl_scores in (
            _pg_bm25_messages(conn, msg_phrase, limit, logger, candidate_ids, user_id=user_id),
            _pg_bm25_attachments(conn, att_phrase, limit, logger, candidate_ids, user_id=user_id),
        ):
            for mid, raw in tbl_scores.items():
                boosted = raw * 1.5
                if mid not in scores or boosted > scores[mid]:
                    scores[mid] = boosted

    # Strategy 2: Individual terms (any word matches — broader recall).
    for tbl_scores in (
        _pg_bm25_messages(conn, msg_disj, limit, logger, candidate_ids, user_id=user_id),
        _pg_bm25_attachments(conn, att_disj, limit, logger, candidate_ids, user_id=user_id),
    ):
        for mid, raw in tbl_scores.items():
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw

    # Normalize to 0-1 (single result gets 1.0).
    if scores:
        max_score = max(scores.values())
        min_score = min(scores.values())
        if max_score == min_score:
            scores = {mid: 1.0 for mid in scores}
        else:
            score_range = max_score - min_score
            scores = {mid: (s - min_score) / score_range for mid, s in scores.items()}

    return scores
