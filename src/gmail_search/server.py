import logging
import re
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Query  # noqa: F401  (Body used in POST /api/sql)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from gmail_search.search.engine import SearchEngine
from gmail_search.store.cost import check_budget, get_total_spend
from gmail_search.store.db import _pg_dsn, get_connection
from gmail_search.store.queries import get_attachments_for_message, get_message

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Read-only SQL endpoint helpers
# ─────────────────────────────────────────────────────────────────
SQL_MAX_ROWS = 500
SQL_MAX_QUERY_CHARS = 5000
SQL_TIMEOUT_SEC = 10.0

# Keywords that cannot appear in a `/api/sql` query. The SQLite-specific
# entries (PRAGMA, LOAD_EXTENSION, READFILE, WRITEFILE, EDIT,
# FTS3_TOKENIZER, SQLITE_*) are kept as-is — Postgres parses them as
# identifiers, so blocking them remains harmless and guards against any
# future backend-switch slip-up.
_SQL_FORBIDDEN = re.compile(
    r"\b(ATTACH|DETACH|PRAGMA|VACUUM|ANALYZE|REINDEX|INSERT|UPDATE|"
    r"DELETE|DROP|CREATE|REPLACE|TRUNCATE|ALTER|BEGIN|COMMIT|ROLLBACK|"
    r"SAVEPOINT|RELEASE|LOAD_EXTENSION|READFILE|WRITEFILE|EDIT|COPY|"
    r"GRANT|REVOKE|SET|RESET|SHOW|LISTEN|NOTIFY|DISCARD|LOCK|CLUSTER|"
    r"FTS3_TOKENIZER|SQLITE_COMPILEOPTION_GET|SQLITE_COMPILEOPTION_USED|"
    r"SQLITE_SOURCE_ID|SQLITE_VERSION|SQLITE_LOG)\b",
    re.IGNORECASE,
)

# Block introspection of backend internals. The sqlite_* names are
# legacy guards; on Postgres the real leaks are `pg_catalog.*` and
# `information_schema.*`, which expose role/password metadata, role
# membership, and (via pg_stat_activity) any other session's query
# text. `/api/sql_schema` surfaces the documented table shape; these
# are not needed.
_SQL_BLOCKED_TABLES = re.compile(
    r"\b(sqlite_master|sqlite_schema|sqlite_temp_master|sqlite_temp_schema|"
    r"sqlite_sequence|sqlite_stat1|sqlite_stat4|sqlite_dbpage|sqlite_dbstat|"
    r"pg_catalog|pg_user|pg_shadow|pg_authid|pg_roles|pg_stat_activity|"
    r"pg_settings|pg_hba_file_rules|information_schema)\b",
    re.IGNORECASE,
)

# Negative LIMIT bypasses our row cap (SQLite treats it as "no limit").
_SQL_NEGATIVE_LIMIT = re.compile(r"\bLIMIT\s*-\s*\d", re.IGNORECASE)

_SQL_STARTS_WITH_SELECT_OR_WITH = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

_SQL_LINE_COMMENT = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _open_readonly_connection():
    """Open a Postgres connection in a READ ONLY transaction with a
    statement-level timeout. Two-layer defense: the gate validator
    rejects non-SELECT queries, and the server refuses writes at the
    transaction level.
    """
    import psycopg

    conn = psycopg.connect(_pg_dsn(), autocommit=False)
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        # Belt-and-suspenders timeout at the server. PG's SET LOCAL
        # doesn't support bound params — we inline an int we control
        # ourselves, so there's no injection surface.
        timeout_ms = int(SQL_TIMEOUT_SEC * 1000)
        cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
    return conn


def _strip_for_validation(query: str) -> str:
    """Strip comments + string literals so keyword checks see real tokens.

    SQLite tolerates `IN/* */SERT`-style keyword splitting. We strip block
    + line comments before pattern matching so an attacker can't smuggle a
    forbidden keyword past the regex by interleaving comments. Read-only
    mode at the connection level is still the primary defense.
    """
    stripped = _SQL_BLOCK_COMMENT.sub(" ", query)
    stripped = _SQL_LINE_COMMENT.sub(" ", stripped)
    stripped = re.sub(r"'([^'\\]|\\.)*'", "''", stripped)
    stripped = re.sub(r'"([^"\\]|\\.)*"', '""', stripped)
    return stripped


def _validate_sql(query: str) -> str | None:
    """Return None if the query passes the safety gate, else an error string."""
    if not query or not query.strip():
        return "query required"
    if len(query) > SQL_MAX_QUERY_CHARS:
        return f"query too long (>{SQL_MAX_QUERY_CHARS} chars)"
    if not _SQL_STARTS_WITH_SELECT_OR_WITH.match(query):
        return "only SELECT / WITH queries are allowed"
    stripped = _strip_for_validation(query)
    hit = _SQL_FORBIDDEN.search(stripped)
    if hit:
        return f"disallowed SQL keyword: {hit.group(0).upper()}"
    blocked = _SQL_BLOCKED_TABLES.search(stripped)
    if blocked:
        return f"introspection of {blocked.group(0)} is not allowed; use /api/sql_schema"
    if _SQL_NEGATIVE_LIMIT.search(stripped):
        return "negative LIMIT is not allowed"
    if ";" in stripped.rstrip().rstrip(";"):
        return "multiple statements are not allowed"
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (bytes, memoryview)):
        return f"<blob {len(value)} bytes>"
    return value


def _run_sql_with_timeout(db_path: Path, query: str) -> dict:
    """Execute a SELECT and capture the first SQL_MAX_ROWS rows.

    Enforces the timeout at the PG server via `statement_timeout`, set
    on the transaction by `_open_readonly_connection`. The transaction
    is also READ ONLY, so even a gate bypass cannot mutate state.
    `db_path` is ignored; kept for call-site compatibility.
    """
    conn = _open_readonly_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            columns = [d[0] for d in cursor.description or []]
            rows_raw = cursor.fetchmany(SQL_MAX_ROWS + 1)
        truncated = len(rows_raw) > SQL_MAX_ROWS
        rows = [[_json_safe(v) for v in row] for row in rows_raw[:SQL_MAX_ROWS]]
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
        }
    finally:
        conn.close()


def _build_query_filters(
    sender: str | None,
    subject_contains: str | None,
    date_from: str | None,
    date_to: str | None,
    label: str | None,
) -> tuple[list[str], list]:
    clauses: list[str] = []
    params: list = []
    if sender:
        clauses.append("m.from_addr LIKE %s")
        params.append(f"%{sender}%")
    if subject_contains:
        clauses.append("m.subject LIKE %s")
        params.append(f"%{subject_contains}%")
    if date_from:
        clauses.append("m.date >= %s")
        params.append(date_from)
    if date_to:
        clauses.append("m.date <= %s")
        params.append(f"{date_to}T23:59:59+00:00")
    if label:
        clauses.append("m.labels LIKE %s")
        params.append(f'%"{label}"%')
    return clauses, params


def _thread_ids_matching_filters(
    conn,
    clauses: list[str],
    params: list,
    has_attachment: bool | None,
    order_by: str,
    limit: int,
) -> list[str]:
    where = " AND ".join(clauses) if clauses else "1=1"
    join = ""
    if has_attachment is True:
        join = "INNER JOIN attachments a ON a.message_id = m.id"
    elif has_attachment is False:
        where += " AND m.id NOT IN (SELECT message_id FROM attachments)"
    order = "MAX(m.date) DESC" if order_by == "date_desc" else "MAX(m.date) ASC"
    sql = f"""SELECT m.thread_id, MAX(m.date) as last_date
             FROM messages m {join}
             WHERE {where}
             GROUP BY m.thread_id
             ORDER BY {order}
             LIMIT %s"""
    rows = conn.execute(sql, [*params, limit]).fetchall()
    return [r["thread_id"] for r in rows]


def _load_thread_summaries(conn, thread_ids: list[str]) -> list[dict]:
    if not thread_ids:
        return []
    import json as _json

    placeholders = ",".join(["%s"] * len(thread_ids))
    rows = conn.execute(
        f"""SELECT thread_id, subject, participants, message_count,
            date_first, date_last
            FROM thread_summary WHERE thread_id IN ({placeholders})""",
        thread_ids,
    ).fetchall()
    by_id = {r["thread_id"]: r for r in rows}
    latest_snippets = _latest_snippet_per_thread(conn, thread_ids)
    out = []
    for tid in thread_ids:
        r = by_id.get(tid)
        if r is None:
            continue
        out.append(
            {
                "thread_id": r["thread_id"],
                "subject": r["subject"],
                "participants": _json.loads(r["participants"]),
                "message_count": r["message_count"],
                "date_first": r["date_first"],
                "date_last": r["date_last"],
                "snippet": latest_snippets.get(tid, ""),
            }
        )
    return out


def _latest_snippet_per_thread(conn, thread_ids: list[str]) -> dict[str, str]:
    if not thread_ids:
        return {}
    placeholders = ",".join(["%s"] * len(thread_ids))
    rows = conn.execute(
        f"""SELECT m.thread_id, m.body_text FROM messages m
            INNER JOIN (
                SELECT thread_id, MAX(date) as max_date
                FROM messages WHERE thread_id IN ({placeholders})
                GROUP BY thread_id
            ) latest ON m.thread_id = latest.thread_id AND m.date = latest.max_date""",
        thread_ids,
    ).fetchall()
    return {r["thread_id"]: (r["body_text"] or "")[:500] for r in rows}


def _run_structured_query(
    db_path: Path,
    sender: str | None,
    subject_contains: str | None,
    date_from: str | None,
    date_to: str | None,
    label: str | None,
    has_attachment: bool | None,
    order_by: str,
    limit: int,
) -> list[dict]:
    conn = get_connection(db_path)
    try:
        clauses, params = _build_query_filters(sender, subject_contains, date_from, date_to, label)
        thread_ids = _thread_ids_matching_filters(conn, clauses, params, has_attachment, order_by, limit)
        return _load_thread_summaries(conn, thread_ids)
    finally:
        conn.close()


def create_app(
    db_path: Path,
    data_dir: Path,
    config: dict[str, Any],
) -> FastAPI:
    # Fail fast at startup if a developer added a table to SCHEMA without
    # documenting it in TABLE_DOCS — the LLM would silently miss it.
    from gmail_search.store.db import assert_table_docs_cover_schema, get_connection, reap_stale_jobs

    assert_table_docs_cover_schema()

    # Sweep stale `running` job_progress rows left by crashed workers so
    # /api/status never surfaces a zombie banner on a fresh boot. Pure DB
    # — no process inspection here (that's the `gmail-search reap` CLI).
    _conn = get_connection(db_path)
    try:
        reap_stale_jobs(_conn)
    finally:
        _conn.close()

    app = FastAPI(title="Gmail Search")

    templates_dir = Path(__file__).parent.parent.parent / "templates"
    # Index dir is resolved on-demand in `get_engine` / `_prewarm_engine`
    # / `_engine_swap_watcher` so a mid-reindex pointer flip is picked
    # up without a server restart (see below). We don't bind it at
    # server-init time — the dir we saw at boot can be GC'd by the
    # next reindex cycle, stranding any cached reference.
    from gmail_search.index.searcher import resolve_active_index_dir

    def _format_thread_result(r):
        return {
            "thread_id": r.thread_id,
            "score": r.score,
            "similarity": r.similarity,
            "subject": r.subject,
            "participants": r.participants,
            "message_count": r.message_count,
            "date_first": r.date_first,
            "date_last": r.date_last,
            "user_replied": r.user_replied,
            "matches": [
                {
                    "message_id": m.message_id,
                    "score": m.score,
                    "from_addr": m.from_addr,
                    "date": m.date,
                    "snippet": m.snippet,
                    "match_type": m.match_type,
                    "attachment_filename": m.attachment_filename,
                }
                for m in r.matches
            ],
        }

    def _collect_result_message_ids(results):
        ids = set()
        for r in results:
            for m in r.matches:
                ids.add(m.message_id)
        return ids

    def _compute_topic_facets(results, msg_topics):
        """Count how many result threads fall into each leaf topic.

        A thread belongs to a topic if any of its matching messages do.
        Counts threads (not messages) so facet counts match the result list.
        """
        from collections import Counter

        topic_thread_counts: Counter = Counter()
        topic_labels: dict[str, str] = {}

        for r in results:
            thread_topics = set()
            for m in r.matches:
                for tid in msg_topics.get(m.message_id, []):
                    thread_topics.add(tid)
            for tid in thread_topics:
                topic_thread_counts[tid] += 1

        # Get labels for the topics we found
        if topic_thread_counts:
            conn_f = get_connection(db_path)
            placeholders = ",".join(["%s"] * len(topic_thread_counts))
            rows = conn_f.execute(
                f"SELECT topic_id, label FROM topics WHERE topic_id IN ({placeholders})",
                list(topic_thread_counts.keys()),
            ).fetchall()
            conn_f.close()
            topic_labels = {r["topic_id"]: r["label"] for r in rows}

        return sorted(
            [
                {"topic_id": tid, "label": topic_labels.get(tid, tid), "count": count}
                for tid, count in topic_thread_counts.items()
            ],
            key=lambda f: f["count"],
            reverse=True,
        )

    # Engine state. Guarded by `_engine_lock` because both the startup
    # coroutine and the background poller can try to rebuild at the
    # same time, and the search handler reads from the same slot.
    import threading as _threading

    _engine: SearchEngine | None = None
    _engine_index_dir: Path | None = None
    _engine_lock = _threading.Lock()

    def _build_engine(path: Path) -> SearchEngine:
        """Synchronous construction — SymSpell dict load + sharded
        ScaNN load runs here. Called from startup (once) and from the
        background poller (on pointer flip). The search request path
        NEVER builds synchronously; it always reads the cached
        instance via `get_engine()`.
        """
        return SearchEngine(db_path, path, config)

    def get_engine() -> SearchEngine:
        """Return the currently-warm SearchEngine.

        Reads `_engine` under the lock. If the engine was somehow not
        warmed yet (e.g. a request raced startup), falls back to a
        synchronous build so the request still succeeds — but the
        happy path is "already built, just return it". The background
        poller swaps `_engine` atomically when the index pointer
        moves.
        """
        nonlocal _engine, _engine_index_dir
        with _engine_lock:
            if _engine is not None:
                return _engine
        # Miss — startup hasn't finished yet. Fall back to a sync
        # build so the request doesn't fail. This path shouldn't hit
        # in practice once startup completes.
        current = resolve_active_index_dir(db_path, data_dir / "scann_index")
        built = _build_engine(current)
        with _engine_lock:
            if _engine is None:
                _engine = built
                _engine_index_dir = current
            return _engine  # another thread may have won the race — return whoever's in the slot

    async def _prewarm_engine() -> None:
        """Startup hook: build the initial engine in a worker thread so
        uvicorn's event loop doesn't block, and log how long it took
        so the operator can tell if SymSpell / ScaNN is pathological.
        """
        import asyncio as _asyncio
        import time as _time

        nonlocal _engine, _engine_index_dir
        start = _time.time()
        current = resolve_active_index_dir(db_path, data_dir / "scann_index")
        built = await _asyncio.to_thread(_build_engine, current)
        with _engine_lock:
            _engine = built
            _engine_index_dir = current
        logger.info(f"engine prewarmed in {_time.time() - start:.1f}s (index={current.name})")

    async def _engine_swap_watcher() -> None:
        """Background task: poll the DB pointer every 10s. When it
        moves (a reindex landed), build the replacement engine off
        the event loop, then atomically swap. The old engine keeps
        serving until the new one is ready, so the swap is invisible
        to the user — no 10-30s SymSpell stall on the first post-
        reindex search.
        """
        import asyncio as _asyncio
        import time as _time

        nonlocal _engine, _engine_index_dir
        while True:
            try:
                await _asyncio.sleep(10)
                current = resolve_active_index_dir(db_path, data_dir / "scann_index")
                with _engine_lock:
                    stale = _engine_index_dir is not None and current != _engine_index_dir
                if not stale:
                    continue
                start = _time.time()
                built = await _asyncio.to_thread(_build_engine, current)
                with _engine_lock:
                    _engine = built
                    _engine_index_dir = current
                logger.info(f"engine hot-swapped in {_time.time() - start:.1f}s (index={current.name})")
            except _asyncio.CancelledError:
                raise
            except Exception as e:
                # Don't let the watcher die — worst case we fall back
                # to the lazy rebuild in get_engine() next search.
                logger.warning(f"engine swap watcher error: {e}")

    @app.on_event("startup")
    async def _on_startup() -> None:  # noqa: RUF029
        import asyncio as _asyncio

        # Block startup on the initial build so the first incoming
        # request hits a warm engine, not a cold one.
        await _prewarm_engine()
        # Then fire off the poller as a background task.
        _asyncio.create_task(_engine_swap_watcher())

    @app.get("/", response_class=HTMLResponse)
    async def home():
        html_file = templates_dir / "index.html"
        return html_file.read_text()

    def _lookup_message_topics(msg_ids):
        """Map message IDs to their leaf topic IDs for client-side filtering."""
        if not msg_ids:
            return {}
        conn_t = get_connection(db_path)
        placeholders = ",".join(["%s"] * len(msg_ids))
        rows = conn_t.execute(
            f"""SELECT mt.message_id, mt.topic_id FROM message_topics mt
                JOIN topics t ON mt.topic_id = t.topic_id
                WHERE mt.message_id IN ({placeholders})
                AND t.topic_id NOT IN (SELECT DISTINCT parent_id FROM topics WHERE parent_id IS NOT NULL)""",
            list(msg_ids),
        ).fetchall()
        conn_t.close()
        result = {}
        for r in rows:
            result.setdefault(r["message_id"], []).append(r["topic_id"])
        return result

    @app.get("/api/search")
    async def api_search(
        q: str = Query(..., min_length=1, max_length=1000),
        k: int = Query(20, ge=1, le=100),
        sort: str = Query("relevance", pattern="^(relevance|recent)$"),
        filter: bool = Query(True, alias="filter"),
        date_from: str | None = Query(None, description="ISO date YYYY-MM-DD (inclusive)", max_length=32),
        date_to: str | None = Query(None, description="ISO date YYYY-MM-DD (inclusive)", max_length=32),
    ):
        from gmail_search.summarize import get_summaries_bulk_meta

        engine = get_engine()
        results = engine.search_threads(
            q,
            top_k=k,
            sort=sort,
            filter_offtopic=filter,
            date_from=date_from,
            date_to=date_to,
        )

        # Look up topic IDs for all result messages (for client-side filtering)
        all_msg_ids = _collect_result_message_ids(results)
        msg_topics = _lookup_message_topics(all_msg_ids)

        # Pre-computed per-message summaries — lets the agent answer many
        # questions without fetching full thread bodies via get_thread.
        # We pull the freshest row regardless of model/prompt version
        # AND surface the key + created_at so the UI debug panel can
        # show where each summary came from.
        conn_s = get_connection(db_path)
        summary_meta = get_summaries_bulk_meta(conn_s, all_msg_ids)
        conn_s.close()

        facets = _compute_topic_facets(results, msg_topics)

        # Tag each result with its topic IDs + attach match-level summaries.
        formatted = []
        for r in results:
            fr = _format_thread_result(r)
            topics = set()
            for m in r.matches:
                topics.update(msg_topics.get(m.message_id, []))
            fr["topic_ids"] = list(topics)
            for match in fr["matches"]:
                meta = summary_meta.get(match["message_id"])
                match["summary"] = (meta or {}).get("summary", "") or ""
                match["summary_model"] = (meta or {}).get("model")
                match["summary_created_at"] = (meta or {}).get("created_at")
            formatted.append(fr)

        return {
            "results": formatted,
            "facets": facets,
        }

    @app.get("/api/query")
    async def api_query(
        sender: str | None = Query(None, description="Substring match on from_addr"),
        subject_contains: str | None = Query(None),
        date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
        date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
        label: str | None = Query(None, description="Gmail label, e.g. INBOX, IMPORTANT"),
        has_attachment: bool | None = Query(None),
        order_by: str = Query("date_desc", pattern="^(date_desc|date_asc)$"),
        limit: int = Query(20, le=100),
    ):
        threads = _run_structured_query(
            db_path,
            sender=sender,
            subject_contains=subject_contains,
            date_from=date_from,
            date_to=date_to,
            label=label,
            has_attachment=has_attachment,
            order_by=order_by,
            limit=limit,
        )
        return {"results": threads}

    @app.get("/api/inbox")
    async def api_inbox(
        limit: int = Query(50, le=200),
        offset: int = Query(0, ge=0),
    ):
        """Priority inbox: threads carrying both the IMPORTANT and INBOX
        labels, newest-first, grouped by thread. Matches Gmail's own
        "Priority Inbox" definition (Important + in the inbox).
        """
        import json as _json

        conn = get_connection(db_path)
        try:
            # One query, one round-trip. Previous version fired four
            # (thread-id list → thread_summary bulk → latest-message-id
            # bulk → summaries bulk). CTEs:
            #   priority_threads — IMPORTANT+INBOX threads, paginated.
            #                      Sort by `date_last` so the UI order
            #                      matches the date the UI displays.
            #   latest_msg       — DISTINCT ON (thread_id) gives the
            #                      newest-dated message per thread, from
            #                      which we pull id + from_addr + body.
            # Final LEFT JOIN pulls in the latest summary (message_summaries
            # is keyed by message_id, at most one row per message).
            rows = conn.execute(
                """
                WITH priority_threads AS (
                    SELECT ts.thread_id,
                           ts.subject,
                           ts.participants,
                           ts.message_count,
                           ts.date_first,
                           ts.date_last
                    FROM thread_summary ts
                    WHERE EXISTS (
                        SELECT 1 FROM messages m
                        WHERE m.thread_id = ts.thread_id
                          AND m.labels LIKE %s
                          AND m.labels LIKE %s
                    )
                    ORDER BY ts.date_last DESC
                    LIMIT %s OFFSET %s
                ),
                latest_msg AS (
                    SELECT DISTINCT ON (m.thread_id)
                           m.thread_id,
                           m.id AS latest_message_id,
                           m.from_addr AS latest_from_addr,
                           m.body_text AS latest_body
                    FROM messages m
                    WHERE m.thread_id IN (SELECT thread_id FROM priority_threads)
                    ORDER BY m.thread_id, m.date DESC
                )
                SELECT pt.thread_id,
                       pt.subject,
                       pt.participants,
                       pt.message_count,
                       pt.date_first,
                       pt.date_last,
                       lm.latest_message_id,
                       lm.latest_from_addr,
                       lm.latest_body,
                       ms.summary,
                       ms.model       AS summary_model,
                       ms.created_at  AS summary_created_at
                FROM priority_threads pt
                JOIN latest_msg lm USING (thread_id)
                LEFT JOIN message_summaries ms
                       ON ms.message_id = lm.latest_message_id
                ORDER BY pt.date_last DESC
                """,
                ('%"IMPORTANT"%', '%"INBOX"%', limit, offset),
            ).fetchall()

            results = []
            for r in rows:
                row: dict = {
                    "thread_id": r["thread_id"],
                    "subject": r["subject"],
                    "participants": _json.loads(r["participants"]),
                    "message_count": r["message_count"],
                    "date_first": r["date_first"],
                    "date_last": r["date_last"],
                    "snippet": (r["latest_body"] or "")[:500],
                    "latest_message_id": r["latest_message_id"],
                    "latest_from_addr": r["latest_from_addr"],
                }
                if r["summary"]:
                    row["summary"] = r["summary"]
                    row["summary_model"] = r["summary_model"]
                    row["summary_created_at"] = (
                        r["summary_created_at"].isoformat()
                        if hasattr(r["summary_created_at"], "isoformat")
                        else r["summary_created_at"]
                    )
                results.append(row)
            return {"results": results}
        finally:
            conn.close()

    @app.get("/api/thread/{thread_id}")
    async def api_thread(thread_id: str):
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT id FROM messages WHERE thread_id = %s ORDER BY date",
            (thread_id,),
        ).fetchall()
        messages = []
        for row in rows:
            msg = get_message(conn, row["id"])
            if msg is None:
                continue
            attachments = get_attachments_for_message(conn, msg.id)
            messages.append(
                {
                    "id": msg.id,
                    "from_addr": msg.from_addr,
                    "to_addr": msg.to_addr,
                    "subject": msg.subject,
                    "body_text": msg.body_text,
                    "date": msg.date.isoformat(),
                    "labels": msg.labels,
                    "attachments": [
                        {
                            "id": a.id,
                            "filename": a.filename,
                            "mime_type": a.mime_type,
                            "size_bytes": a.size_bytes,
                        }
                        for a in attachments
                    ],
                }
            )
        conn.close()
        return {"thread_id": thread_id, "messages": messages}

    @app.get("/api/topics")
    async def api_topics():
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT topic_id, parent_id, label, depth, message_count, top_senders FROM topics ORDER BY depth, message_count DESC"
        ).fetchall()
        conn.close()
        import json as _json

        return [
            {
                "topic_id": r["topic_id"],
                "parent_id": r["parent_id"],
                "label": r["label"],
                "depth": r["depth"],
                "message_count": r["message_count"],
                "top_senders": _json.loads(r["top_senders"]),
            }
            for r in rows
        ]

    @app.get("/api/message/{message_id}")
    async def api_message(message_id: str):
        conn = get_connection(db_path)
        msg = get_message(conn, message_id)
        if msg is None:
            conn.close()
            return JSONResponse({"error": "Message not found"}, status_code=404)
        attachments = get_attachments_for_message(conn, message_id)
        conn.close()
        return {
            "id": msg.id,
            "thread_id": msg.thread_id,
            "from_addr": msg.from_addr,
            "to_addr": msg.to_addr,
            "subject": msg.subject,
            "body_text": msg.body_text,
            "date": msg.date.isoformat(),
            "labels": msg.labels,
            "attachments": [
                {
                    "id": a.id,
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "size_bytes": a.size_bytes,
                }
                for a in attachments
            ],
        }

    @app.get("/api/thread_lookup")
    async def api_thread_lookup(cite_ref: str = Query(..., min_length=4, max_length=20)):
        """Resolve a cite_ref (4-20 char prefix) to a real thread_id.

        Returns the thread + subject when exactly one thread starts with
        the prefix. 404 if zero matches; 409 if ambiguous.
        """
        prefix = cite_ref.strip().lower()
        if not prefix or any(c not in "0123456789abcdef" for c in prefix):
            return JSONResponse({"error": "cite_ref must be hex"}, status_code=400)
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT thread_id, subject FROM thread_summary WHERE thread_id LIKE %s LIMIT 5",
            (f"{prefix}%",),
        ).fetchall()
        conn.close()
        if not rows:
            return JSONResponse({"error": f"no thread starts with {prefix!r}"}, status_code=404)
        if len(rows) > 1:
            return JSONResponse(
                {
                    "error": f"{len(rows)} threads start with {prefix!r} — be more specific",
                    "candidates": [{"thread_id": r["thread_id"], "subject": r["subject"]} for r in rows],
                },
                status_code=409,
            )
        return {"thread_id": rows[0]["thread_id"], "subject": rows[0]["subject"]}

    @app.post("/api/sql")
    async def api_sql(payload: dict = Body(...)):
        """Run a read-only SQL SELECT (or WITH...SELECT) against the DB.

        Hard limits: max 500 rows returned, 10s timeout, 5000 char query.
        Enforced read-only at the SQLite connection level AND via keyword
        blacklist (defense in depth). Statement must begin with SELECT or
        WITH; multiple statements are rejected.
        """
        import psycopg

        query = str(payload.get("query", ""))
        err = _validate_sql(query)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        try:
            return _run_sql_with_timeout(db_path, query)
        except psycopg.errors.QueryCanceled as e:
            # statement_timeout tripped — surface as 408 so the UI can
            # tell timeout apart from a syntax error.
            return JSONResponse({"error": f"SQL timeout: {e!s}"}, status_code=408)
        except psycopg.Error as e:
            return JSONResponse({"error": f"SQL error: {e!s}"}, status_code=400)

    @app.post("/api/battle/vote")
    async def api_battle_vote(payload: dict = Body(...)):
        """Record a model battle outcome.

        Body: {question, variant_a, variant_b, winner, request_id_a?, request_id_b?}
        where winner ∈ {"a","b","tie","both_bad"} and variant_{a,b} are
        JSON objects like {"model":"...","thinkingLevel":"..."}.
        """
        import json as _json

        question = str(payload.get("question", "")).strip()
        va = payload.get("variant_a")
        vb = payload.get("variant_b")
        winner = str(payload.get("winner", ""))
        if winner not in {"a", "b", "tie", "both_bad"}:
            return JSONResponse({"error": "winner must be a|b|tie|both_bad"}, status_code=400)
        if not isinstance(va, dict) or not isinstance(vb, dict):
            return JSONResponse({"error": "variant_a and variant_b must be objects"}, status_code=400)
        conn = get_connection(db_path)
        # RETURNING id works on SQLite 3.35+ and Postgres; replaces the
        # old cursor.lastrowid which returned None under psycopg's
        # dict_row factory.
        cur = conn.execute(
            """INSERT INTO model_battles
                 (question, variant_a, variant_b, winner, request_id_a, request_id_b)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                question[:1000],
                _json.dumps(va),
                _json.dumps(vb),
                winner,
                payload.get("request_id_a"),
                payload.get("request_id_b"),
            ),
        )
        row = cur.fetchone()
        conn.commit()
        if row is None:
            row_id = 0
        else:
            try:
                row_id = int(row["id"])
            except (KeyError, TypeError):
                row_id = int(row[0])
        conn.close()
        return {"ok": True, "id": row_id}

    @app.get("/api/battle/stats")
    async def api_battle_stats():
        """Per-variant win rate + head-to-head matrix."""
        import json as _json

        conn = get_connection(db_path)
        rows = conn.execute("SELECT variant_a, variant_b, winner FROM model_battles").fetchall()
        conn.close()

        def key(v: dict) -> str:
            return f"{v.get('model','?')} · {v.get('thinkingLevel','?')}"

        wins: dict[str, int] = {}
        losses: dict[str, int] = {}
        ties: dict[str, int] = {}
        both_bad: dict[str, int] = {}
        for r in rows:
            a = key(_json.loads(r["variant_a"]))
            b = key(_json.loads(r["variant_b"]))
            for v in (a, b):
                wins.setdefault(v, 0)
                losses.setdefault(v, 0)
                ties.setdefault(v, 0)
                both_bad.setdefault(v, 0)
            w = r["winner"]
            if w == "a":
                wins[a] += 1
                losses[b] += 1
            elif w == "b":
                wins[b] += 1
                losses[a] += 1
            elif w == "tie":
                ties[a] += 1
                ties[b] += 1
            else:  # both_bad
                both_bad[a] += 1
                both_bad[b] += 1

        leaderboard = []
        for v in sorted(wins.keys()):
            total = wins[v] + losses[v] + ties[v] + both_bad[v]
            win_rate = wins[v] / total if total else 0.0
            leaderboard.append(
                {
                    "variant": v,
                    "wins": wins[v],
                    "losses": losses[v],
                    "ties": ties[v],
                    "both_bad": both_bad[v],
                    "total": total,
                    "win_rate": round(win_rate, 3),
                }
            )
        leaderboard.sort(key=lambda x: (-x["win_rate"], -x["total"]))
        return {"leaderboard": leaderboard, "battles": len(rows)}

    @app.get("/api/conversations")
    async def api_conversations_list(limit: int = Query(100, le=500)):
        conn = get_connection(db_path)
        rows = conn.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM conversation_messages m WHERE m.conversation_id = c.id) as message_count
               FROM conversations c
               ORDER BY c.updated_at DESC
               LIMIT %s""",
            (limit,),
        ).fetchall()
        conn.close()
        return {
            "conversations": [
                {
                    "id": r["id"],
                    "title": r["title"] or "New chat",
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": r["message_count"],
                }
                for r in rows
            ]
        }

    @app.get("/api/conversations/{conversation_id}")
    async def api_conversation_get(conversation_id: str):
        import json as _json

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = %s",
            (conversation_id,),
        ).fetchone()
        if row is None:
            conn.close()
            return JSONResponse({"error": "not found"}, status_code=404)
        msg_rows = conn.execute(
            """SELECT seq, role, parts FROM conversation_messages
               WHERE conversation_id = %s ORDER BY seq""",
            (conversation_id,),
        ).fetchall()
        conn.close()
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": [{"seq": m["seq"], "role": m["role"], "parts": _json.loads(m["parts"])} for m in msg_rows],
        }

    @app.put("/api/conversations/{conversation_id}")
    async def api_conversation_save(conversation_id: str, payload: dict = Body(...)):
        """Upsert conversation + fully replace its message list.

        Body: {title?: str, messages: [{role, parts}]}
        """
        import json as _json

        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            return JSONResponse({"error": "messages must be a list"}, status_code=400)
        title = payload.get("title")
        conn = get_connection(db_path)
        try:
            now = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
            conn.execute(
                """INSERT INTO conversations (id, title, created_at, updated_at)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT(id) DO UPDATE SET
                     title = COALESCE(excluded.title, conversations.title),
                     updated_at = excluded.updated_at""",
                (conversation_id, title, now, now),
            )
            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = %s",
                (conversation_id,),
            )
            for seq, m in enumerate(messages):
                conn.execute(
                    """INSERT INTO conversation_messages (conversation_id, seq, role, parts)
                       VALUES (%s, %s, %s, %s)""",
                    (
                        conversation_id,
                        seq,
                        str(m.get("role", "user")),
                        _json.dumps(m.get("parts", [])),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "id": conversation_id}

    @app.delete("/api/conversations/{conversation_id}")
    async def api_conversation_delete(conversation_id: str):
        conn = get_connection(db_path)
        conn.execute("DELETE FROM conversation_messages WHERE conversation_id = %s", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
        conn.commit()
        conn.close()
        return {"ok": True}

    @app.get("/api/attachment/{attachment_id}/text")
    async def api_attachment_text(attachment_id: int):
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT filename, mime_type, extracted_text FROM attachments WHERE id = %s",
            (attachment_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return JSONResponse({"error": "Attachment not found"}, status_code=404)
        return {
            "attachment_id": attachment_id,
            "filename": row["filename"],
            "mime_type": row["mime_type"],
            "extracted_text": row["extracted_text"] or "",
        }

    @app.get("/api/attachment/{attachment_id}")
    async def api_attachment(attachment_id: int):
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT raw_path, mime_type, filename FROM attachments WHERE id = %s",
            (attachment_id,),
        ).fetchone()
        conn.close()
        if row is None or not row["raw_path"]:
            return JSONResponse({"error": "Attachment not found"}, status_code=404)
        # Validate path is under data_dir to prevent path traversal
        resolved = Path(row["raw_path"]).resolve()
        if not str(resolved).startswith(str(data_dir.resolve())):
            return JSONResponse({"error": "Invalid attachment path"}, status_code=403)
        if not resolved.exists():
            return JSONResponse({"error": "Attachment file missing"}, status_code=404)
        return FileResponse(
            str(resolved),
            media_type=row["mime_type"],
            filename=row["filename"],
        )

    @app.get("/api/progress")
    async def api_progress():
        from gmail_search.store.db import JobProgress

        return JobProgress.get(db_path) or []

    @app.get("/api/sql_schema")
    async def api_sql_schema():
        # Markdown description of every queryable table — surfaced to the
        # chat LLM so the sql_query tool knows the real column shapes.
        from gmail_search.store.db import describe_schema_for_llm

        return {"markdown": describe_schema_for_llm()}

    @app.get("/api/status")
    async def api_status():
        from gmail_search.store.db import JobProgress

        conn = get_connection(db_path)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        # All dates are now UTC so string sort works correctly
        dates = conn.execute("SELECT MIN(date) as oldest, MAX(date) as newest FROM messages").fetchone()
        total_cost = get_total_spend(conn)
        ok, spent, remaining = check_budget(conn, config["budget"]["max_usd"])
        # Search-side cost: query_cache tells us how many distinct
        # search embeds we've paid for; the costs table (operation=
        # 'embed_query') tells us how much they totalled. Both are
        # tiny at our volume but worth surfacing so users can see
        # search-as-you-type isn't silently expensive.
        query_count_row = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()
        query_cost_row = conn.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM costs WHERE operation = 'embed_query'"
        ).fetchone()
        conn.close()
        jobs = JobProgress.get(db_path) or []
        running = [j for j in jobs if j["status"] == "running"]
        return {
            "messages": msg_count,
            "embeddings": emb_count,
            "date_oldest": dates["oldest"],
            "date_newest": dates["newest"],
            "total_cost_usd": round(total_cost, 4),
            "budget_remaining_usd": round(remaining, 4),
            "query_embeds": int(query_count_row[0] or 0),
            "query_embed_cost_usd": round(float(query_cost_row[0] or 0.0), 6),
            "running_job": running[0] if running else None,
        }

    # ── Background-job plumbing (frontfill + backfill share everything) ──
    #
    # Both are long-running gmail-search subcommands spawned detached from
    # the HTTP layer. Authoritative status is a PID file under data_dir;
    # start writes it, stop SIGTERMs + unlinks it, status reads it. The
    # two endpoints below are just (pid_name, argv) configuration — the
    # shared helpers handle everything else.
    #
    # PID files:
    #   data/watch.pid     frontfill (the watch daemon)
    #   data/backfill.pid  backfill (the update pipeline)

    def _pid_file_status(pid_path: Path) -> dict:
        """Is the process at pid_path alive AND not a zombie?

        os.kill(pid, 0) returns success for zombies (they still exist
        in the process table after exiting, waiting to be reaped), so
        we use psutil to distinguish zombie from live. When we spot
        one, we also waitpid() to reap it if the server is the parent,
        so the process table doesn't accumulate corpses.
        """
        import os

        import psutil  # inline — formatter strips unused top-level imports

        if not pid_path.exists():
            return {"running": False, "pid": None}

        def _clean(pid: int | None = None):
            pid_path.unlink(missing_ok=True)
            if pid is not None:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, OSError):
                    pass
            return {"running": False, "pid": None}

        try:
            pid = int(pid_path.read_text().strip())
            proc = psutil.Process(pid)
            if proc.status() == psutil.STATUS_ZOMBIE:
                return _clean(pid)
            return {"running": True, "pid": pid}
        except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
            return _clean()

    def _start_detached_job(pid_filename: str, log_filename: str, extra_args: list[str]) -> dict:
        """Spawn `gmail-search <extra_args> --data-dir <data_dir>` detached,
        write pid to data_dir/pid_filename, return ok payload. 409 if the
        pid file already points at a live process.
        """
        from fastapi.responses import JSONResponse

        from gmail_search.jobs import gmail_search_command, spawn_detached

        pid_path = data_dir / pid_filename
        status = _pid_file_status(pid_path)
        if status["running"]:
            return JSONResponse(
                status_code=409,
                content={"ok": False, "error": f"already running (pid {status['pid']})"},
            )

        argv = gmail_search_command() + extra_args + ["--data-dir", str(data_dir)]
        pid = spawn_detached(argv, data_dir / log_filename)
        pid_path.write_text(str(pid))
        return {"ok": True, "pid": pid}

    def _stop_detached_job(pid_filename: str):
        """SIGTERM the pid recorded in data_dir/pid_filename and unlink
        the file. Idempotent: 404 when not running, otherwise 200 whether
        or not the process was still alive (the daemon's signal handler
        does the clean job_progress finish).
        """
        import os
        import signal

        from fastapi.responses import JSONResponse

        pid_path = data_dir / pid_filename
        if not pid_path.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "not running"})
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            pid_path.unlink(missing_ok=True)
            return {"ok": True, "pid": pid}
        except (ValueError, ProcessLookupError, PermissionError):
            pid_path.unlink(missing_ok=True)
            return {"ok": True, "detail": "process was already gone"}

    def _enrich_with_eta(job: dict) -> dict:
        """Add `rate_per_sec` + `eta_seconds` fields to a running job
        row when enough signal exists to compute them. Works off the
        `start_completed` baseline we record at job start, so the rate
        is meaningful from the very first poll.
        """
        from datetime import datetime, timezone

        if job["status"] != "running" or job["total"] <= 0:
            return job
        baseline = job.get("start_completed") or 0
        delta = job["completed"] - baseline
        if delta <= 0:
            return job
        try:
            started = datetime.fromisoformat(job["started_at"])
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        except (ValueError, TypeError):
            return job
        if elapsed < 5:  # too little signal early on
            return job
        rate = delta / elapsed
        remaining = max(0, job["total"] - job["completed"])
        return {
            **job,
            "rate_per_sec": round(rate, 3),
            "eta_seconds": round(remaining / rate) if rate > 0 else None,
        }

    @app.get("/api/jobs/running")
    async def api_jobs_running():
        """Everything /settings needs: all running job_progress rows +
        disk usage + per-job (frontfill/backfill) authoritative status
        from the pid files. Running rows are enriched with rate_per_sec
        + eta_seconds when the baseline + elapsed time permit it.
        """
        import shutil as _shutil

        from gmail_search.store.db import JobProgress

        jobs = JobProgress.get(db_path) or []
        enriched = [_enrich_with_eta(j) for j in jobs]
        running = [j for j in enriched if j["status"] == "running"]
        usage = _shutil.disk_usage(str(data_dir))
        return {
            "running": running,
            "recent": enriched[:5],
            "disk": {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            },
            "frontfill": _pid_file_status(data_dir / "watch.pid"),
            "backfill": _pid_file_status(data_dir / "backfill.pid"),
            "summarize": _pid_file_status(data_dir / "summarize.pid"),
        }

    # ── OAuth status + re-auth ──────────────────────────────────────────
    # Settings UI reads /api/auth/status to show which scopes the current
    # token grants, and POSTs /api/auth/reauth to force a consent-screen
    # re-prompt (needed after we expand SCOPES in gmail/auth.py). Re-auth
    # spawns `gmail-search auth --force` detached so the browser flow
    # runs outside the server process; the token file lands under data/
    # and the status endpoint reflects it on next poll.

    _SCOPE_LABELS = {
        "https://www.googleapis.com/auth/gmail.readonly": "gmail",
        "https://www.googleapis.com/auth/drive.readonly": "drive",
    }

    @app.get("/api/auth/status")
    async def api_auth_status():
        """Return which scopes the stored token actually grants, plus
        a friendly `missing` list against the SCOPES our code expects.
        """
        import json

        from gmail_search.gmail.auth import SCOPES

        token_path = data_dir / "token.json"
        granted: list[str] = []
        if token_path.exists():
            try:
                tok = json.loads(token_path.read_text())
                granted = list(tok.get("scopes") or [])
            except Exception:
                granted = []
        expected = set(SCOPES)
        missing = sorted(expected - set(granted))
        granted_sorted = sorted(granted)
        return {
            "token_present": token_path.exists(),
            "granted": granted_sorted,
            "granted_labels": [_SCOPE_LABELS.get(s, s) for s in granted_sorted],
            "missing": missing,
            "missing_labels": [_SCOPE_LABELS.get(s, s) for s in missing],
            "drive_enabled": "https://www.googleapis.com/auth/drive.readonly" in granted,
        }

    @app.post("/api/auth/reauth")
    async def api_auth_reauth():
        """Kick a detached `gmail-search auth --force` so the OAuth
        flow opens in the user's default browser. Same pid-file pattern
        as the other jobs — lets the UI show running/done state.
        """
        return _start_detached_job(
            pid_filename="auth.pid",
            log_filename="auth.log",
            extra_args=["auth", "--force"],
        )

    @app.post("/api/jobs/frontfill")
    async def api_jobs_frontfill(interval: int = Query(120, ge=10, le=86400)):
        """Start the continuous watch daemon: sync new messages every
        `interval` seconds, then extract/embed/reindex. Writes
        data/watch.pid, same convention as `gmail-search start`, so CLI
        and UI stay interchangeable.
        """
        return _start_detached_job(
            pid_filename="watch.pid",
            log_filename="watch.log",
            extra_args=["watch", "--interval", str(interval)],
        )

    @app.post("/api/jobs/frontfill/stop")
    async def api_jobs_frontfill_stop():
        return _stop_detached_job("watch.pid")

    @app.post("/api/jobs/backfill")
    async def api_jobs_backfill(min_free_gb: float = Query(5.0, ge=0.1, le=10000.0)):
        """Pull older messages through the full pipeline. Runs forever
        via `--loop`: once caught up (or if a cycle crashes), sleeps 5
        min and resumes. Pauses batches when free disk < min_free_gb.
        """
        return _start_detached_job(
            pid_filename="backfill.pid",
            log_filename="backfill.log",
            extra_args=["update", "--min-free-gb", str(min_free_gb), "--loop"],
        )

    @app.post("/api/jobs/backfill/stop")
    async def api_jobs_backfill_stop():
        return _stop_detached_job("backfill.pid")

    @app.post("/api/jobs/summarize")
    async def api_jobs_summarize(
        concurrency: int = Query(12, ge=1, le=64),
        batch_size: int = Query(1, ge=1, le=16),
    ):
        """Run `gmail-search summarize` in the background. Rate + ETA
        land in /api/jobs/running via the `summarize` job_progress row.
        """
        return _start_detached_job(
            pid_filename="summarize.pid",
            log_filename="summarize.log",
            extra_args=[
                "summarize",
                "--concurrency",
                str(concurrency),
                "--batch-size",
                str(batch_size),
            ],
        )

    @app.post("/api/jobs/summarize/stop")
    async def api_jobs_summarize_stop():
        return _stop_detached_job("summarize.pid")

    return app
