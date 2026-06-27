import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Query  # noqa: F401  (Body used in POST /api/sql)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from gmail_search.auth import require_user_id
from gmail_search.search.engine import SearchEngine
from gmail_search.store.cost import check_budget, get_total_spend
from gmail_search.store.db import _pg_dsn, get_connection
from gmail_search.store.queries import extract_attachment_on_demand, get_attachments_for_message, get_message

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Read-only SQL endpoint helpers
# ─────────────────────────────────────────────────────────────────
SQL_MAX_ROWS = 500
SQL_MAX_QUERY_CHARS = 5000
# Per-query statement_timeout for the read-only /api/sql path. These are
# analytical queries the agent generates (BM25 search + text extraction over
# body_text across large match sets), which legitimately run several seconds;
# 10s was too tight. Tunable via env without a code change.
try:
    SQL_TIMEOUT_SEC = float(os.environ.get("GMAIL_SQL_TIMEOUT_SEC", "30.0"))
except ValueError:
    SQL_TIMEOUT_SEC = 30.0
# Reject footgun values: 0 DISABLES Postgres' statement_timeout (unbounded
# queries), and NaN/inf would later break int(SQL_TIMEOUT_SEC * 1000). Fall
# back to the default and clamp to a sane [1, 120]s window.
if SQL_TIMEOUT_SEC != SQL_TIMEOUT_SEC or SQL_TIMEOUT_SEC <= 0 or SQL_TIMEOUT_SEC > 120:
    SQL_TIMEOUT_SEC = 30.0

# How long to keep a hot-swapped-out ScaNN searcher alive before closing it.
# Long enough for any in-flight query that captured the old searcher to finish
# (queries are sub-second), short enough that we don't keep a full ~3-4 GB
# index copy resident between swaps (which land ~15 min apart). Tunable.
RETIRED_SEARCHER_GRACE_SECONDS = 5.0

# Conversation IDs are interpolated into filesystem paths
# (`deep-conv-<id>`) and used as PKs in `conversation_claude_session`.
# Strict allow-list at the API boundary so downstream code can trust
# the value. Mirrors sandbox.py:_safe_conversation_id.
_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Claude session UUIDs (claudebox `sessionId`) are interpolated into
# filesystem paths in the `/jsonl` endpoint. The pg_schema column
# is plain TEXT with no format check, and the value comes from
# parsing claudebox's response — so we constrain it at read time.
# UUID4 hex shape: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
_CLAUDE_SESSION_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

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

# BM25-required columns. These are the columns covered by
# `messages_bm25_idx` and `attachments_bm25_idx`. `LIKE`/`ILIKE`
# against any of them forces a parallel seq scan over ~410k
# messages or ~600k attachments (observed: 1-2s/query, ~50x slower
# than the BM25 path). The agent JSONLs showed Anthropic-side
# agents reverting to LIKE despite system-prompt guidance, so we
# reject server-side and surface the BM25 translation in the
# error string. Escape hatch: `@@@` anywhere in the query — once
# the agent has BM25-prefiltered, a follow-up LIKE on a small row
# set is fine.
_SQL_BM25_REQUIRED = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
    r"(subject|body_text|from_addr|to_addr|filename|extracted_text)\b"
    r"\s*(?:NOT\s+)?I?LIKE\b",
    re.IGNORECASE,
)

# Negative LIMIT bypasses our row cap (SQLite treats it as "no limit").
_SQL_NEGATIVE_LIMIT = re.compile(r"\bLIMIT\s*-\s*\d", re.IGNORECASE)

_SQL_STARTS_WITH_SELECT_OR_WITH = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

_SQL_LINE_COMMENT = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _open_readonly_connection(*, user_id: str | None = None):
    """Open a Postgres connection in a READ ONLY transaction with a
    statement-level timeout. When `user_id` is given (the multi-tenant
    /api/sql path), additionally drops to the non-superuser
    `gmail_search_reader` role and binds `app.user_id` so the RLS
    policies installed by pg_schema.sql scope every read to that user.

    Three-layer defense:
      1. The gate validator rejects non-SELECT queries (regex level).
      2. The transaction is READ ONLY at the PG transaction level.
      3. With `user_id` set, RLS policies on every per-user table
         (messages, attachments, conversations, …) filter rows to
         `WHERE user_id = current_setting('app.user_id')` — so even
         `SELECT count(*) FROM messages` returns only the active
         user's count.

    Daemon callers can pass `user_id=None` and stay on the superuser
    role (BYPASSRLS), so the watch/update/summarize loops are
    unaffected.
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
        if user_id is not None:
            # Drop privileges to the reader role for the duration of
            # this transaction. The reader has SELECT only on
            # LLM-facing tables and is NOT BYPASSRLS, so the policy
            # checks below take effect. After ROLLBACK the connection
            # is back to the gmail_search role for whatever runs next.
            cur.execute("SET LOCAL ROLE gmail_search_reader")
            # `set_config(setting, value, is_local)` is the only form
            # that accepts a parameterized value — `SET LOCAL …` is
            # parsed as raw SQL and rejects %s. is_local=true scopes
            # the setting to this transaction only.
            cur.execute(
                "SELECT set_config('app.user_id', %s, true)",
                (user_id,),
            )
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
    # BM25 enforcement runs last so the cheaper structural checks
    # short-circuit first. Skip if the query already uses `@@@` —
    # that means the agent has BM25-prefiltered and a follow-up
    # LIKE on a small row set is legitimate.
    if "@@@" not in stripped:
        bm25 = _SQL_BM25_REQUIRED.search(stripped)
        if bm25:
            col = bm25.group(1).lower()
            return _bm25_required_error(col)
    return None


def _bm25_required_error(column: str) -> str:
    """Format the rejection message with a concrete BM25 translation
    for the column the agent tried to LIKE-filter on. Goal: the next
    tool call after this error should be a correct BM25 query — no
    re-prompting needed."""
    table = "attachments" if column in ("filename", "extracted_text") else "messages"
    return (
        f"`{column} LIKE/ILIKE '%...%'` is rejected — it forces a seq scan on "
        f"the {table} table. Use the ParadeDB BM25 index instead "
        f"(~50x faster). Translate:\n"
        f"  WHERE {column} ILIKE '%foo%'\n"
        f"    →  WHERE id @@@ '{column}:foo'\n"
        f"  WHERE from_addr LIKE '%delta%' AND subject LIKE '%credit%'\n"
        f"    →  WHERE id @@@ 'from_addr:delta AND subject:credit'\n"
        f"  WHERE body_text LIKE '%refund issued%'  (multi-word phrase)\n"
        f"    →  WHERE id @@@ 'body_text:\"refund issued\"'\n"
        f"Add `ORDER BY paradedb.score(id) DESC` for relevance ranking. "
        f"Escape hatch: include `@@@` anywhere (e.g. an `id @@@ '...'` "
        f"prefilter) and the LIKE check is skipped — use that only when "
        f"BM25 truly cannot express the predicate. Call `describe_schema` "
        f"if unsure which columns are BM25-indexed."
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (bytes, memoryview)):
        return f"<blob {len(value)} bytes>"
    return value


def _run_sql_with_timeout(db_path: Path, query: str, *, user_id: str | None = None) -> dict:
    """Execute a SELECT and capture the first SQL_MAX_ROWS rows.

    Enforces the timeout at the PG server via `statement_timeout`, set
    on the transaction by `_open_readonly_connection`. The transaction
    is also READ ONLY, so even a gate bypass cannot mutate state.
    `db_path` is ignored; kept for call-site compatibility.

    When `user_id` is given, the transaction additionally runs as the
    `gmail_search_reader` role with `app.user_id` bound — RLS policies
    then scope every per-user table read to that user. This is the
    /api/sql path. Internal callers can omit user_id to stay on the
    superuser role.
    """
    conn = _open_readonly_connection(user_id=user_id)
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


def _load_thread_summaries(conn, thread_ids: list[str], *, user_id: str) -> list[dict]:
    if not thread_ids:
        return []
    import json as _json

    placeholders = ",".join(["%s"] * len(thread_ids))
    rows = conn.execute(
        f"""SELECT thread_id, subject, participants, message_count,
            date_first, date_last
            FROM thread_summary WHERE thread_id IN ({placeholders})
              AND user_id = %s""",
        thread_ids + [user_id],
    ).fetchall()
    by_id = {r["thread_id"]: r for r in rows}
    latest_snippets = _latest_snippet_per_thread(conn, thread_ids, user_id=user_id)
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


def _latest_snippet_per_thread(conn, thread_ids: list[str], *, user_id: str) -> dict[str, str]:
    if not thread_ids:
        return {}
    placeholders = ",".join(["%s"] * len(thread_ids))
    rows = conn.execute(
        f"""SELECT m.thread_id, m.body_text FROM messages m
            INNER JOIN (
                SELECT thread_id, MAX(date) as max_date
                FROM messages
                WHERE user_id = %s AND thread_id IN ({placeholders})
                GROUP BY thread_id
            ) latest ON m.thread_id = latest.thread_id AND m.date = latest.max_date
            WHERE m.user_id = %s""",
        [user_id] + thread_ids + [user_id],
    ).fetchall()
    return {r["thread_id"]: (r["body_text"] or "")[:500] for r in rows}


# Manifest inline-eligibility threshold in DECODED BYTES. The tool
# layer's cap is in BASE64 CHARS (see MAX_INLINE_BYTES in
# web/lib/tools.ts) which is ~1.33× bigger than the decoded payload,
# so using the same numeric value here gives a slightly conservative
# manifest (true "can_inline" only when comfortably under the wire
# budget). Bump both together if the cap ever changes.
_INLINE_BYTES_CAP = 15 * 1024 * 1024


def _attachment_manifest_dict(a) -> dict:
    """Build the attachment row the model (and the UI) sees.

    Adds a capability manifest on top of the raw DB columns so the agent
    can choose a representation without guessing: `text_chars` advertises
    how much extracted text we already have, `can_inline_pdf` /
    `can_inline_image` reflect the ship-bytes size budget, and
    `suggested_as` is a server recommendation — still the model's choice.

    Input is an `Attachment` dataclass (every current call site constructs
    one via `get_attachments_for_message`); kept narrow on purpose so a
    future dict-shaped row doesn't silently mis-render.
    """
    mime = getattr(a, "mime_type", "") or ""
    size = int(getattr(a, "size_bytes", 0) or 0)
    text = getattr(a, "extracted_text", "") or ""
    text_chars = len(text)
    is_pdf = mime == "application/pdf"
    is_image = mime.startswith("image/")
    fits_inline = size > 0 and size <= _INLINE_BYTES_CAP
    can_inline_pdf = is_pdf and fits_inline
    can_inline_image = is_image and fits_inline
    can_render_pages = is_pdf  # page rasterization path works regardless of size
    # Decision tree: if we already have "enough" text (500 chars ≈ a short
    # paragraph is roughly the threshold where text is usually useful),
    # prefer text. Otherwise prefer the binary path appropriate to the
    # mime type. Fall back to "text" so the agent at least tries
    # on-demand extraction.
    if text_chars >= 500:
        suggested = "text"
    elif can_inline_image:
        suggested = "inline_image"
    elif can_inline_pdf:
        suggested = "inline_pdf"
    elif can_render_pages:
        suggested = "rendered_pages"
    else:
        suggested = "text"
    return {
        "id": getattr(a, "id", None),
        "filename": getattr(a, "filename", None),
        "mime_type": mime,
        "size_bytes": size,
        "text_chars": text_chars,
        "can_inline_pdf": can_inline_pdf,
        "can_inline_image": can_inline_image,
        "can_render_pages": can_render_pages,
        "suggested_as": suggested,
    }


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
    *,
    user_id: str,
) -> list[dict]:
    conn = get_connection(db_path)
    try:
        clauses, params = _build_query_filters(sender, subject_contains, date_from, date_to, label)
        # Always scope by user_id — _thread_ids_matching_filters is the
        # gate for the structured /api/query endpoint, and we don't
        # want a from:foo filter to surface another user's threads.
        clauses.append("m.user_id = %s")
        params.append(user_id)
        thread_ids = _thread_ids_matching_filters(conn, clauses, params, has_attachment, order_by, limit)
        return _load_thread_summaries(conn, thread_ids, user_id=user_id)
    finally:
        conn.close()


def _inbox_rows(
    conn,
    predicate_sql: str,
    predicate_params: tuple,
    limit: int,
    offset: int,
    *,
    user_id: str | None = None,
) -> list[dict]:
    """Return inbox-shaped thread rows whose messages match `predicate_sql`.

    `predicate_sql` is spliced into a parameterised `EXISTS` subquery — it
    must be a trusted, statically-defined fragment (never user input);
    every value inside it must be a `%s` placeholder bound via
    `predicate_params`. The built query is the same one that used to live
    inside `api_inbox`: take the newest `limit` thread_summary rows where
    the predicate matches, then join the latest message + its summary.

    `user_id` scopes everything so a label-based predicate doesn't leak
    rows from another user's INBOX/STARRED/etc. None falls back to the
    bootstrap user (legacy single-pool callers + tests that pre-date
    the multi-tenant gating).
    """
    import json as _json

    if user_id is None:
        from gmail_search.auth.write_user import get_bootstrap_user_id

        user_id = get_bootstrap_user_id(conn)

    sql = f"""
        WITH priority_threads AS (
            SELECT ts.thread_id,
                   ts.subject,
                   ts.participants,
                   ts.message_count,
                   ts.date_first,
                   ts.date_last
            FROM thread_summary ts
            WHERE ts.user_id = %s AND EXISTS (
                SELECT 1 FROM messages m
                WHERE m.thread_id = ts.thread_id
                  AND m.user_id = %s
                  AND ({predicate_sql})
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
            WHERE m.user_id = %s
              AND m.thread_id IN (SELECT thread_id FROM priority_threads)
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
    """
    # Bind order: ts.user_id, m.user_id (in EXISTS), predicate params, limit, offset, m.user_id (latest_msg).
    rows = conn.execute(sql, (user_id, user_id, *predicate_params, limit, offset, user_id)).fetchall()

    results: list[dict] = []
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
    return results


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
    # Stash the data dir on app.state so the auth dependency (and
    # anything else added later) can resolve gmail_search.db without
    # a global. Phase 1 of PER_USER_LOGIN.
    app.state.data_dir = data_dir

    # Mount the opt-in deep-analysis agent surface. See
    # docs/DEEP_ANALYSIS_AGENT.md. Chat mode is unaffected; the agent
    # endpoints sit under /api/agent/* and /api/artifact/*.
    from gmail_search.agents.service import register_agent_routes

    register_agent_routes(app, db_path)

    # Auth surface for multi-tenant Phase 1. The endpoints register
    # unconditionally so the bootstrap flow works even when
    # GMAIL_MULTI_TENANT=0; gating happens inside `require_user`.
    from gmail_search.auth.routes import register_auth_routes

    register_auth_routes(app, db_path)

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

    # Engine state — now PER-USER. Each user has their own SearchEngine
    # (own ScaNN index, own contact_freq, own term_aliases, own spell
    # dict) lazy-loaded on first request. Per the 0c benchmark this
    # costs ~129 MiB RSS and ~185 ms cold load per user.
    import threading as _threading

    _engines: dict[str, SearchEngine] = {}
    _engine_index_dirs: dict[str, Path] = {}
    _engine_lock = _threading.Lock()
    # Strong refs to in-flight retired-searcher close tasks. asyncio only holds
    # weak refs to tasks, so without this an unreferenced close task can be GC'd
    # mid-sleep — silently leaking the index copy we meant to free.
    _pending_closes: set = set()

    def _user_index_dir(user_id: str) -> Path:
        # Per-user fallback dir; resolve_active_index_dir uses the DB
        # pointer first and only returns this if there's no pointer row.
        return data_dir / "users" / user_id / "scann_index"

    def _build_engine(user_id: str, path: Path) -> SearchEngine:
        """Synchronous construction — SymSpell dict load + sharded
        ScaNN load runs here. Called lazily on first request per user
        and again whenever the per-user index pointer moves (rebuild
        watcher). The search request path NEVER builds synchronously
        on the happy path — once cached, it's a dict lookup."""
        return SearchEngine(db_path, path, config, user_id=user_id)

    def get_engine(user_id: str) -> SearchEngine:
        """Return (or lazy-build) the SearchEngine for this user.

        Cold path is ~185ms (per 0c benchmark) the first time a user
        hits search. Subsequent requests are dict lookups."""
        with _engine_lock:
            cached = _engines.get(user_id)
            if cached is not None:
                return cached
        current = resolve_active_index_dir(db_path, _user_index_dir(user_id), user_id=user_id)
        built = _build_engine(user_id, current)
        with _engine_lock:
            # Re-check under lock — another thread may have built it.
            cached = _engines.get(user_id)
            if cached is not None:
                return cached
            _engines[user_id] = built
            _engine_index_dirs[user_id] = current
            return built

    async def _prewarm_engine() -> None:
        """Startup hook: prewarm the bootstrap user's engine so the
        operator's first search isn't a 185ms cold load. Other users
        get lazy-loaded on first request."""
        import asyncio as _asyncio
        import time as _time

        from gmail_search.auth.write_user import get_bootstrap_user_id
        from gmail_search.store.db import get_connection as _get_conn

        try:
            _conn = _get_conn(db_path)
            try:
                bootstrap_uid = get_bootstrap_user_id(_conn)
            finally:
                _conn.close()
        except Exception as e:
            logger.warning(f"engine prewarm: no bootstrap user yet ({e}); deferring")
            return
        start = _time.time()
        current = resolve_active_index_dir(db_path, _user_index_dir(bootstrap_uid), user_id=bootstrap_uid)
        built = await _asyncio.to_thread(_build_engine, bootstrap_uid, current)
        with _engine_lock:
            _engines[bootstrap_uid] = built
            _engine_index_dirs[bootstrap_uid] = current
        logger.info(f"engine prewarmed for {bootstrap_uid} in {_time.time() - start:.1f}s (index={current.name})")

    def _schedule_retired_searcher_close(old_searcher: object, user_id: str) -> None:
        """Free a hot-swapped-out searcher promptly after a short grace delay.

        The atomic swap is already done, so new queries use the new searcher;
        only requests that captured the OLD searcher before the swap still hold
        it. Those are sub-second, so we wait RETIRED_SEARCHER_GRACE_SECONDS to
        let them drain, then close in a thread (the C++ teardown can block).
        Closing promptly — rather than holding the retired copy until the next
        swap ~15 min later — reclaims ~3-4 GB of steady-state RSS."""
        import asyncio as _asyncio

        async def _close_after_grace() -> None:
            try:
                await _asyncio.sleep(RETIRED_SEARCHER_GRACE_SECONDS)
                await _asyncio.to_thread(old_searcher.close)
            except _asyncio.CancelledError:
                # Server shutting down — close inline so we don't leak the C++
                # index, then re-raise to honor the cancellation.
                try:
                    old_searcher.close()
                except Exception:
                    pass
                raise
            except Exception as e:
                logger.warning(f"retired searcher close failed for {user_id}: {e}")

        task = _asyncio.create_task(_close_after_grace())
        _pending_closes.add(task)
        task.add_done_callback(_pending_closes.discard)

    async def _engine_swap_watcher() -> None:
        """Background task: poll every cached user's index pointer.
        When one moves (a reindex landed for that user), build the
        replacement engine off the event loop, then atomically swap.
        Old engine keeps serving until the new one is ready."""
        import asyncio as _asyncio
        import time as _time

        while True:
            try:
                await _asyncio.sleep(10)
                with _engine_lock:
                    cached_users = list(_engine_index_dirs.items())
                for user_id, prev_dir in cached_users:
                    current = resolve_active_index_dir(db_path, _user_index_dir(user_id), user_id=user_id)
                    if current == prev_dir:
                        continue
                    start = _time.time()
                    engine = _engines.get(user_id)
                    if engine is None:
                        continue  # not built yet — the lazy get_engine path will pick up `current`
                    # Reload ONLY the ScaNN searcher (the sole thing a light
                    # reindex changes), reusing the engine's spell dict / contact
                    # map / aliases. The old _build_engine reconstructed the whole
                    # engine — ~80s of GIL-bound reloads that froze serve every
                    # swap. reload_index is mostly GIL-releasing C++.
                    old_searcher = await _asyncio.to_thread(engine.reload_index, current)
                    with _engine_lock:
                        _engine_index_dirs[user_id] = current
                    # Free the retired searcher promptly after a grace delay
                    # (instead of holding a full ~3-4 GB index copy resident
                    # until the next swap) — see _schedule_retired_searcher_close.
                    if old_searcher is not None:
                        _schedule_retired_searcher_close(old_searcher, user_id)
                    logger.info(
                        f"engine index hot-swapped for {user_id} in {_time.time() - start:.1f}s "
                        f"(index={current.name})"
                    )
            except _asyncio.CancelledError:
                raise
            except Exception as e:
                # Don't let the watcher die — worst case the next
                # lazy-build path catches up.
                logger.warning(f"engine swap watcher error: {e}")

    @app.on_event("startup")
    async def _on_startup() -> None:  # noqa: RUF029
        import asyncio as _asyncio

        from gmail_search.claudebox_creds import sync_credentials_if_stale

        # Sync host Claude OAuth creds into the claudebox bind-mount
        # before serving any traffic. This is the most common cause
        # of the "Failed to authenticate" 401 surfacing inside deep-
        # mode model output — the host refreshes on its own, the
        # mount lags. The check is also re-run per-turn in
        # service.py:_real_run as belt-and-suspenders.
        sync_credentials_if_stale()
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
        user_id: str = Depends(require_user_id),
    ):
        from gmail_search.summarize import get_summaries_bulk_meta

        try:
            engine = get_engine(user_id)
        except FileNotFoundError as e:
            # New user without a built ScaNN index yet — return clean
            # empty results (with a `pending` flag the UI can surface)
            # instead of 500ing. Backfill builds the index on its first
            # reindex pass; this is the window before that completes.
            logger.info("search before index build for user %s: %s", user_id, e)
            return {
                "results": [],
                "facets": [],
                "pending_index": True,
            }
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

    @app.get("/api/find_facts")
    async def api_find_facts(
        q: str = Query(..., min_length=1, max_length=1000),
        exhaustive: bool = Query(True),
        k: int = Query(200, ge=1, le=500),
        hybrid: bool = Query(True),
        collapse: bool = Query(False),
        user_id: str = Depends(require_user_id),
    ):
        """Enumerate atomic facts (propositions) matching `q` across this
        user's mailbox via hybrid (semantic ∪ keyword) retrieval. `k`
        caps the result count. Defensive: creates the propositions table
        + BM25 index if absent, so this works before the proposition
        backfill has run (returning {"facts": []} on an empty table)."""
        from gmail_search import propositions

        conn = get_connection(db_path)
        try:
            propositions.ensure_table(conn)
            propositions.ensure_bm25_index(conn)
            try:
                embedder = getattr(get_engine(user_id), "embedder", None)
            except FileNotFoundError:
                # New user without a built ScaNN index yet — fall through
                # to a fresh embedder rather than 500ing.
                embedder = None
            if embedder is None:
                from gmail_search.config import load_config
                from gmail_search.embed.client import GeminiEmbedder

                embedder = GeminiEmbedder(load_config())
            facts = propositions.find_facts(
                conn,
                embedder,
                user_id=user_id,
                query=q,
                exhaustive=exhaustive,
                cap=k,
                hybrid=hybrid,
                collapse_near_dups=collapse,
            )
        finally:
            conn.close()
        return {"facts": facts}

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
        user_id: str = Depends(require_user_id),
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
            user_id=user_id,
        )
        return {"results": threads}

    @app.get("/api/inbox")
    async def api_inbox(
        limit: int = Query(50, le=200),
        offset: int = Query(0, ge=0),
        user_id: str = Depends(require_user_id),
    ):
        """Inbox view. Mirrors Gmail's default web INBOX listing —
        every thread with at least one message carrying the INBOX
        label, newest-first. (Gmail's classic all-in-one view, NOT
        the category-tab split — Promotions / Updates / Social stay
        in the list.)
        """
        conn = get_connection(db_path)
        try:
            # Thread is "in the inbox" if ANY message on it still has
            # the INBOX label. We check per-message labels because
            # archiving removes INBOX from one message but leaves it
            # on earlier ones; matching on messages (not
            # thread_summary.all_labels) catches that drift.
            results = _inbox_rows(
                conn,
                "m.labels LIKE %s",
                ('%"INBOX"%',),
                limit,
                offset,
                user_id=user_id,
            )
            return {"results": results}
        finally:
            conn.close()

    @app.get("/api/priority-inbox")
    async def api_priority_inbox(
        limit: int = Query(25, le=100),
        offset: int = Query(0, ge=0),
        user_id: str = Depends(require_user_id),
    ):
        """Gmail-style Priority Inbox — three sections:

          1. Important and unread — IMPORTANT + UNREAD + INBOX
          2. Starred             — STARRED (regardless of INBOX / read)
          3. Everything else     — INBOX, minus anything already in
                                    sections 1 or 2

        Each section is independently paginated (same limit/offset per
        section) and sorted newest-first by `thread_summary.date_last`.
        Rows use the exact same shape as `/api/inbox` — the frontend
        can feed them through the same adapter.
        """
        conn = get_connection(db_path)
        try:
            important_unread = _inbox_rows(
                conn,
                "m.labels LIKE %s AND m.labels LIKE %s AND m.labels LIKE %s",
                ('%"IMPORTANT"%', '%"UNREAD"%', '%"INBOX"%'),
                limit,
                offset,
                user_id=user_id,
            )
            starred = _inbox_rows(
                conn,
                "m.labels LIKE %s",
                ('%"STARRED"%',),
                limit,
                offset,
                user_id=user_id,
            )
            # "Everything else" = INBOX threads NOT already in sections
            # 1 or 2. We exclude by thread_id (not message_id) because a
            # section's thread should appear exactly once on the page.
            everything_else = _inbox_rows(
                conn,
                (
                    "m.labels LIKE %s"
                    " AND m.thread_id NOT IN ("
                    "   SELECT m2.thread_id FROM messages m2"
                    "   WHERE m2.user_id = m.user_id"
                    "     AND ("
                    "       (m2.labels LIKE %s AND m2.labels LIKE %s AND m2.labels LIKE %s)"
                    "       OR (m2.labels LIKE %s)"
                    "     )"
                    " )"
                ),
                (
                    '%"INBOX"%',
                    '%"IMPORTANT"%',
                    '%"UNREAD"%',
                    '%"INBOX"%',
                    '%"STARRED"%',
                ),
                limit,
                offset,
                user_id=user_id,
            )
            return {
                "sections": [
                    {
                        "title": "Important and unread",
                        "key": "important_unread",
                        "threads": important_unread,
                    },
                    {
                        "title": "Starred",
                        "key": "starred",
                        "threads": starred,
                    },
                    {
                        "title": "Everything else",
                        "key": "everything_else",
                        "threads": everything_else,
                    },
                ]
            }
        finally:
            conn.close()

    @app.get("/api/thread/{thread_id}")
    async def api_thread(thread_id: str, user_id: str = Depends(require_user_id)):
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT id FROM messages WHERE thread_id = %s AND user_id = %s ORDER BY date",
            (thread_id, user_id),
        ).fetchall()
        messages = []
        for row in rows:
            msg = get_message(conn, row["id"], user_id=user_id)
            if msg is None:
                continue
            attachments = get_attachments_for_message(conn, msg.id, user_id=user_id)
            messages.append(
                {
                    "id": msg.id,
                    "from_addr": msg.from_addr,
                    "to_addr": msg.to_addr,
                    "subject": msg.subject,
                    "body_text": msg.body_text,
                    # Full HTML body is included so the thread drawer can
                    # render HTML-only emails (marketing / receipts / many
                    # transactional messages have empty body_text but
                    # well-formed body_html). The client renders this
                    # inside a sandboxed iframe — the UI layer owns the
                    # isolation, the API just hands over the bytes.
                    "body_html": msg.body_html,
                    "date": msg.date.isoformat(),
                    "labels": msg.labels,
                    "attachments": [_attachment_manifest_dict(a) for a in attachments],
                }
            )
        conn.close()
        return {"thread_id": thread_id, "messages": messages}

    @app.get("/api/topics")
    async def api_topics(user_id: str = Depends(require_user_id)):
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT topic_id, parent_id, label, depth, message_count, top_senders FROM topics "
            "WHERE user_id = %s ORDER BY depth, message_count DESC",
            (user_id,),
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
    async def api_message(message_id: str, user_id: str = Depends(require_user_id)):
        conn = get_connection(db_path)
        msg = get_message(conn, message_id, user_id=user_id)
        if msg is None:
            conn.close()
            return JSONResponse({"error": "Message not found"}, status_code=404)
        attachments = get_attachments_for_message(conn, message_id, user_id=user_id)
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
            "attachments": [_attachment_manifest_dict(a) for a in attachments],
        }

    @app.get("/api/thread_lookup")
    async def api_thread_lookup(
        cite_ref: str = Query(..., min_length=4, max_length=20),
        user_id: str = Depends(require_user_id),
    ):
        """Resolve a cite_ref (4-20 char prefix) to a real thread_id.

        Returns the thread + subject when exactly one thread starts with
        the prefix. 404 if zero matches; 409 if ambiguous.
        """
        prefix = cite_ref.strip().lower()
        if not prefix or any(c not in "0123456789abcdef" for c in prefix):
            return JSONResponse({"error": "cite_ref must be hex"}, status_code=400)
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT thread_id, subject FROM thread_summary " "WHERE thread_id LIKE %s AND user_id = %s LIMIT 5",
            (f"{prefix}%", user_id),
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
    async def api_sql(
        payload: dict = Body(...),
        user_id: str = Depends(require_user_id),
    ):
        """Run a read-only SQL SELECT (or WITH...SELECT) against the DB.

        Hard limits: max 500 rows returned, 10s timeout, 5000 char query.
        Enforced read-only at the connection level AND via keyword
        blacklist (defense in depth). Statement must begin with SELECT or
        WITH; multiple statements are rejected.

        Multi-tenant: gated by require_user_id AND scoped by Postgres
        RLS at the storage layer. The connection drops to the
        `gmail_search_reader` role and binds `app.user_id` to the
        caller's user_id before the LLM's query runs, so policies on
        every per-user table (messages, attachments, conversations…)
        filter rows automatically — even an unscoped `SELECT count(*)
        FROM messages` returns only the active user's count.

        The reader role has SELECT only on LLM-facing tables, so an
        attempt to read `query_cache` / `users` / `embeddings` etc.
        raises permission-denied rather than leaking. See pg_schema.sql
        "Phase 3g — Row-Level Security" block.
        """
        import psycopg

        query = str(payload.get("query", ""))
        err = _validate_sql(query)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        try:
            return _run_sql_with_timeout(db_path, query, user_id=user_id)
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
        # dict_row factory. user_id is required by multi-tenant schema.
        from gmail_search.auth.write_user import resolve_write_user_id as _resolve_uid

        uid = _resolve_uid(conn)
        cur = conn.execute(
            """INSERT INTO model_battles
                 (question, variant_a, variant_b, winner, request_id_a, request_id_b, user_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                question[:1000],
                _json.dumps(va),
                _json.dumps(vb),
                winner,
                payload.get("request_id_a"),
                payload.get("request_id_b"),
                uid,
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
    async def api_conversations_list(
        limit: int = Query(100, le=500),
        user_id: str = Depends(require_user_id),
    ):
        conn = get_connection(db_path)
        rows = conn.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM conversation_messages m WHERE m.conversation_id = c.id) as message_count
               FROM conversations c
               WHERE c.user_id = %s
               ORDER BY c.updated_at DESC
               LIMIT %s""",
            (user_id, limit),
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

    @app.get("/api/conversations/live")
    async def api_conversations_live(
        limit: int = Query(100, le=500),
        user_id: str = Depends(require_user_id),
    ):
        """Sidebar feed: every conversation joined to its latest deep
        turn (if any). The UI polls this every few seconds to render a
        live "what's running where" view across concurrent
        conversations. Empty `latest_session` means the conversation
        has no deep turn yet (chat-only).

        IMPORTANT: this route MUST stay declared above
        `GET /api/conversations/{conversation_id}` so FastAPI matches
        the literal `/live` path before treating it as a conv id."""
        conn = get_connection(db_path)
        try:
            rows = conn.execute(
                """WITH latest AS (
                       SELECT DISTINCT ON (conversation_id)
                              conversation_id, id, status, started_at, finished_at
                       FROM agent_sessions
                       WHERE conversation_id IS NOT NULL AND user_id = %s
                       ORDER BY conversation_id, started_at DESC
                   )
                   SELECT c.id, c.title, c.created_at, c.updated_at,
                          ccs.claude_session_uuid,
                          latest.id           AS latest_session_id,
                          latest.status       AS latest_status,
                          latest.started_at   AS latest_started_at,
                          latest.finished_at  AS latest_finished_at,
                          (SELECT COUNT(*) FROM agent_events ev
                           WHERE ev.session_id = latest.id) AS latest_tool_call_count,
                          (SELECT MAX(created_at) FROM agent_events ev
                           WHERE ev.session_id = latest.id) AS latest_last_event_at
                   FROM conversations c
                   LEFT JOIN conversation_claude_session ccs ON ccs.conversation_id = c.id
                   LEFT JOIN latest ON latest.conversation_id = c.id
                   WHERE c.user_id = %s
                   ORDER BY COALESCE(latest.started_at, c.updated_at::timestamptz) DESC
                   LIMIT %s""",
                (user_id, user_id, limit),
            ).fetchall()
            return {
                "conversations": [
                    {
                        "id": r["id"],
                        "title": r["title"] or "New chat",
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "claude_session_uuid": r["claude_session_uuid"],
                        "latest_session": (
                            {
                                "id": r["latest_session_id"],
                                "status": r["latest_status"],
                                "started_at": r["latest_started_at"],
                                "finished_at": r["latest_finished_at"],
                                "tool_call_count": r["latest_tool_call_count"] or 0,
                                "last_event_at": r["latest_last_event_at"],
                            }
                            if r["latest_session_id"] is not None
                            else None
                        ),
                    }
                    for r in rows
                ]
            }
        finally:
            conn.close()

    @app.get("/api/conversations/{conversation_id}")
    async def api_conversation_get(
        conversation_id: str,
        user_id: str = Depends(require_user_id),
    ):
        import json as _json

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations " "WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            conn.close()
            return JSONResponse({"error": "not found"}, status_code=404)
        # conversation_messages joins through conversations.user_id —
        # the existence check above is the gate, so a direct fetch by
        # conversation_id here is safe.
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
    async def api_conversation_save(
        conversation_id: str,
        payload: dict = Body(...),
        user_id: str = Depends(require_user_id),
    ):
        """Upsert conversation + fully replace its message list.

        Body: {title?: str, messages: [{role, parts}]}
        """
        import json as _json

        # Conversation IDs flow into per-conversation claudebox workspace
        # dir names (`deep-conv-<id>`) and into the
        # `conversation_claude_session` mapping. Strict allow-list keeps
        # that filesystem-and-DB use safe AND collision-free vs.
        # sandbox.py:_safe_conversation_id. Reject at the boundary so
        # downstream code can trust the value.
        if not _CONVERSATION_ID_RE.match(conversation_id):
            return JSONResponse(
                {"error": "conversation_id must match ^[A-Za-z0-9_-]{1,64}$"},
                status_code=400,
            )
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            return JSONResponse({"error": "messages must be a list"}, status_code=400)
        title = payload.get("title")
        conn = get_connection(db_path)
        try:
            now = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
            # Use the auth-resolved user_id (multi-tenant: signed-in
            # user; legacy: bootstrap). The conversation rows can only
            # ever belong to the request's owner.
            conn.execute(
                """INSERT INTO conversations (id, title, created_at, updated_at, user_id)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT(id) DO UPDATE SET
                     title = COALESCE(excluded.title, conversations.title),
                     updated_at = excluded.updated_at
                   WHERE conversations.user_id = excluded.user_id""",
                (conversation_id, title, now, now, user_id),
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
    async def api_conversation_delete(
        conversation_id: str,
        user_id: str = Depends(require_user_id),
    ):
        conn = get_connection(db_path)
        # Scope by user_id so user A can't delete user B's conversation.
        # ON DELETE CASCADE on conversation_messages and
        # conversation_claude_session FKs cleans up the rest.
        conn.execute(
            "DELETE FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        )
        conn.commit()
        conn.close()
        # Best-effort filesystem cleanup so a deleted-then-recreated
        # conversation_id doesn't see a stale workspace. The directory
        # may not exist (no deep turn ever ran) — that's fine.
        ws_dir = Path("deploy/claudebox/workspaces") / f"deep-conv-{conversation_id}"
        if ws_dir.is_dir():
            try:
                import shutil as _shutil

                _shutil.rmtree(ws_dir)
            except OSError as exc:
                logger.warning("conversation delete: could not remove workspace %s: %s", ws_dir, exc)
        return {"ok": True}

    @app.get("/api/conversations/{conversation_id}/debug")
    async def api_conversation_debug(
        conversation_id: str,
        session_limit: int = Query(20, le=200),
        user_id: str = Depends(require_user_id),
    ):
        """Per-conversation debug pane payload. Returns sessions
        (turns) with their tool-call timeline + artifacts + workspace
        path + JSONL path. UI renders as a collapsible-by-turn view.

        Heavy fields (workspace listing, raw JSONL contents) are NOT
        inlined here — the UI fetches them from the dedicated
        endpoints below when the user expands a turn."""
        if not _CONVERSATION_ID_RE.match(conversation_id):
            return JSONResponse(
                {"error": "conversation_id must match ^[A-Za-z0-9_-]{1,64}$"},
                status_code=400,
            )
        conn = get_connection(db_path)
        try:
            conv = conn.execute(
                "SELECT id, title FROM conversations WHERE id = %s AND user_id = %s",
                (conversation_id, user_id),
            ).fetchone()
            if conv is None:
                return JSONResponse({"error": "conversation not found"}, status_code=404)
            ccs = conn.execute(
                "SELECT claude_session_uuid FROM conversation_claude_session WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchone()
            claude_uuid = ccs["claude_session_uuid"] if ccs else None
            sessions = conn.execute(
                """SELECT id, mode, question, status, started_at, finished_at, final_answer
                   FROM agent_sessions
                   WHERE conversation_id = %s
                   ORDER BY started_at DESC
                   LIMIT %s""",
                (conversation_id, session_limit),
            ).fetchall()
            session_payloads = []
            for s in sessions:
                events = conn.execute(
                    """SELECT seq, agent_name, kind, payload, created_at
                       FROM agent_events
                       WHERE session_id = %s
                       ORDER BY seq""",
                    (s["id"],),
                ).fetchall()
                artifacts = conn.execute(
                    """SELECT id, name, mime_type, OCTET_LENGTH(data) AS size_bytes, created_at
                       FROM agent_artifacts
                       WHERE session_id = %s
                       ORDER BY created_at""",
                    (s["id"],),
                ).fetchall()
                session_payloads.append(
                    {
                        "id": s["id"],
                        "mode": s["mode"],
                        "question": s["question"],
                        "status": s["status"],
                        "started_at": s["started_at"],
                        "finished_at": s["finished_at"],
                        "final_answer": s["final_answer"],
                        "events": [
                            {
                                "seq": e["seq"],
                                "agent": e["agent_name"],
                                "kind": e["kind"],
                                "payload": e["payload"],
                                "created_at": e["created_at"],
                            }
                            for e in events
                        ],
                        "artifacts": [
                            {
                                "id": a["id"],
                                "name": a["name"],
                                "mime_type": a["mime_type"],
                                "size_bytes": a["size_bytes"],
                                "created_at": a["created_at"],
                            }
                            for a in artifacts
                        ],
                    }
                )
            workspace_dir = Path("deploy/claudebox/workspaces") / f"deep-conv-{conversation_id}"
            return {
                "conversation": {"id": conv["id"], "title": conv["title"]},
                "claude_session_uuid": claude_uuid,
                "workspace_dir": str(workspace_dir) if workspace_dir.is_dir() else None,
                "sessions": session_payloads,
            }
        finally:
            conn.close()

    @app.get("/api/conversations/{conversation_id}/workspace/tree")
    async def api_conversation_workspace_tree(
        conversation_id: str,
        user_id: str = Depends(require_user_id),
    ):
        """Workspace directory listing for the debug pane. Recursive
        but bounded so a runaway agent that wrote 100k files can't
        blow up the response. Each entry: relative path, size, mtime,
        is_dir."""
        if not _CONVERSATION_ID_RE.match(conversation_id):
            return JSONResponse(
                {"error": "conversation_id must match ^[A-Za-z0-9_-]{1,64}$"},
                status_code=400,
            )
        # Ownership gate: user A can't list user B's workspace tree.
        _conn = get_connection(db_path)
        try:
            owner = _conn.execute(
                "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
                (conversation_id, user_id),
            ).fetchone()
        finally:
            _conn.close()
        if owner is None:
            return JSONResponse({"error": "conversation not found"}, status_code=404)
        ws_dir = Path("deploy/claudebox/workspaces") / f"deep-conv-{conversation_id}"
        if not ws_dir.is_dir():
            return JSONResponse({"error": "workspace dir not present"}, status_code=404)
        # Defense-in-depth: even though the regex constrains
        # conversation_id, resolve() the dir + skip symlinks during
        # traversal so a symlink ANY agent could write into the
        # workspace can't make us return metadata for files outside
        # it. We don't follow symlinks (use lstat) and we drop any
        # entry whose resolved path leaves the workspace root.
        ws_root = ws_dir.resolve()
        max_entries = 2000
        entries: list[dict] = []
        for path in ws_dir.rglob("*"):
            if len(entries) >= max_entries:
                break
            try:
                # lstat doesn't follow symlinks — symlink target's
                # metadata is irrelevant; we only describe what's IN
                # the workspace.
                st = path.lstat()
            except OSError:
                continue
            if path.is_symlink():
                # Drop symlinks entirely — listing them is fine in
                # principle but their resolved path could escape; the
                # debug pane has no use for them.
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(ws_root)
            except (OSError, ValueError):
                # Anything that resolves outside the workspace — skip.
                continue
            entries.append(
                {
                    "path": str(path.relative_to(ws_dir)),
                    "is_dir": path.is_dir(),
                    "size_bytes": st.st_size if not path.is_dir() else None,
                    "mtime": st.st_mtime,
                }
            )
        return {
            "workspace_dir": str(ws_dir),
            "entries": entries,
            "truncated": len(entries) >= max_entries,
        }

    @app.get("/api/conversations/{conversation_id}/jsonl")
    async def api_conversation_jsonl(
        conversation_id: str,
        user_id: str = Depends(require_user_id),
    ):
        """Raw Claude Code JSONL transcript for the conversation.
        After Phase 0 confirmed `--resume` appends to the same file,
        the conversation maps to ONE JSONL identified by the pinned
        claude_session_uuid. Streamed as text so the UI can render
        line-by-line without buffering huge transcripts in memory.

        Returns 404 if the workspace or JSONL doesn't exist (no deep
        turn has run yet)."""
        if not _CONVERSATION_ID_RE.match(conversation_id):
            return JSONResponse(
                {"error": "conversation_id must match ^[A-Za-z0-9_-]{1,64}$"},
                status_code=400,
            )
        conn = get_connection(db_path)
        try:
            # Ownership gate first — JOIN through conversations.user_id
            # so user A can't see user B's transcript.
            row = conn.execute(
                """SELECT ccs.claude_session_uuid FROM conversation_claude_session ccs
                   JOIN conversations c ON c.id = ccs.conversation_id
                   WHERE ccs.conversation_id = %s AND c.user_id = %s""",
                (conversation_id, user_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return JSONResponse({"error": "no Claude session pinned to this conversation"}, status_code=404)
        ws_proj_dir = Path("deploy/claudebox/claude-config/projects") / f"-workspaces-deep-conv-{conversation_id}"
        # Defense-in-depth: claude_session_uuid is TEXT in Postgres
        # with no format constraint at the schema level — the value
        # comes from `_extract_session_uuid` parsing claudebox's
        # response, so we don't fully trust it. Reject anything that
        # doesn't look like a UUID (8-4-4-4-12 hex), which is the
        # only shape Claude Code's `--resume` actually produces, AND
        # verify the resolved path stays under ws_proj_dir.
        claude_uuid = str(row["claude_session_uuid"])
        if not _CLAUDE_SESSION_UUID_RE.match(claude_uuid):
            logger.warning(
                "JSONL endpoint refused malformed claude_session_uuid for conv %s: %r",
                conversation_id,
                claude_uuid,
            )
            return JSONResponse({"error": "stored claude_session_uuid has unexpected format"}, status_code=500)
        jsonl_path = ws_proj_dir / f"{claude_uuid}.jsonl"
        try:
            ws_root = ws_proj_dir.resolve()
            jsonl_resolved = jsonl_path.resolve()
            jsonl_resolved.relative_to(ws_root)
        except (OSError, ValueError):
            return JSONResponse({"error": "JSONL path escapes workspace dir"}, status_code=400)
        if not jsonl_resolved.is_file():
            return JSONResponse(
                {"error": f"JSONL not present: {jsonl_resolved}"},
                status_code=404,
            )
        return FileResponse(str(jsonl_resolved), media_type="application/jsonl")

    @app.get("/api/attachment/{attachment_id}/meta")
    async def api_attachment_meta(
        attachment_id: int,
        user_id: str = Depends(require_user_id),
    ):
        """Cheap metadata-only lookup for attachment citation chips.
        Returns enough to render a chip (filename, mime, size) and to
        wire its click to the thread drawer (thread_id, message_id).
        No text / bytes — those have their own endpoints.
        """
        conn = get_connection(db_path)
        try:
            row = conn.execute(
                """SELECT a.id, a.filename, a.mime_type, a.size_bytes,
                          a.message_id, m.thread_id
                   FROM attachments a
                   JOIN messages m ON m.id = a.message_id
                   WHERE a.id = %s AND a.user_id = %s""",
                (attachment_id, user_id),
            ).fetchone()
            if row is None:
                return JSONResponse({"error": "Attachment not found"}, status_code=404)
            return {
                "attachment_id": row["id"],
                "filename": row["filename"],
                "mime_type": row["mime_type"],
                "size_bytes": row["size_bytes"],
                "message_id": row["message_id"],
                "thread_id": row["thread_id"],
            }
        finally:
            conn.close()

    @app.get("/api/attachment/{attachment_id}/text")
    async def api_attachment_text(
        attachment_id: int,
        user_id: str = Depends(require_user_id),
    ):
        conn = get_connection(db_path)
        try:
            row = conn.execute(
                "SELECT filename, mime_type, extracted_text FROM attachments WHERE id = %s AND user_id = %s",
                (attachment_id, user_id),
            ).fetchone()
            if row is None:
                return JSONResponse({"error": "Attachment not found"}, status_code=404)

            text = row["extracted_text"] or ""
            # Fallback: if no extracted_text is stored but the file exists
            # on disk, run the extractor on the fly and persist the result.
            # Bounded by a 20s wall-clock budget so a pathological PDF /
            # OCR run can't hang the endpoint. Any failure degrades
            # silently to the legacy empty-text behaviour.
            if not text.strip():
                text = await _extract_attachment_text_on_demand(conn, attachment_id) or ""

            return {
                "attachment_id": attachment_id,
                "filename": row["filename"],
                "mime_type": row["mime_type"],
                "extracted_text": text,
            }
        finally:
            conn.close()

    async def _extract_attachment_text_on_demand(conn, attachment_id: int) -> str | None:
        """Run on-demand extraction for a single attachment under a 20s
        wall-clock budget. Returns the extracted text or None on any
        failure / timeout / no-op. Side effect: persists the extracted
        text (and image_path) back to the attachments row on success so
        subsequent calls are instant.

        Note: asyncio can't cancel the spawned thread on timeout, so the
        underlying extractor may still run (and commit) after we've
        returned None. That's benign — the persist path is idempotent,
        so at worst the next call sees the now-populated row.
        """

        def _run() -> str | None:
            try:
                result = extract_attachment_on_demand(conn, attachment_id, config=config)
            except Exception:
                logger.exception(f"on-demand extract failed for attachment {attachment_id}")
                return None
            if result is None:
                return None
            return result.text or None

        try:
            return await asyncio.wait_for(asyncio.to_thread(_run), timeout=20.0)
        except asyncio.TimeoutError:
            logger.warning(f"on-demand extract timed out for attachment {attachment_id}")
            return None

    @app.get("/api/attachment/{attachment_id}/render_pages")
    async def api_attachment_render_pages(
        attachment_id: int,
        pages: str = Query(
            "", description="Comma-separated 1-based page numbers, e.g. '1,2,3'. Empty = all pages up to max_pages."
        ),
        max_pages: int = Query(8, ge=1, le=40),
        dpi: int = Query(150, ge=72, le=300),
        user_id: str = Depends(require_user_id),
    ):
        """Rasterize PDF pages to PNG so the model can inline them as
        images. Used by the `get_attachment({as: "rendered_pages"})`
        tool path — the visual option when extracted text isn't rich
        enough and the raw PDF isn't useful to the model (bad parse,
        scan, complex layout).

        Returns `{pages: [{page, base64, mime_type}]}` — base64 PNG
        per page, ordered as requested (or page 1..N if `pages` blank).

        Bounded by a 30s wall-clock budget. Adversarial PDFs
        (decompression bombs, malformed xref) can wedge pymupdf for a
        long time; the timeout protects the request loop.
        """
        # Parse `pages` up-front so a malformed value returns 400 before
        # we touch the DB / filesystem / fitz.
        if pages.strip():
            try:
                requested_raw = [int(p) for p in pages.split(",") if p.strip()]
            except ValueError:
                return JSONResponse({"error": "pages must be comma-separated integers"}, status_code=400)
        else:
            requested_raw = None  # means "first N"

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT raw_path, mime_type FROM attachments WHERE id = %s AND user_id = %s",
            (attachment_id, user_id),
        ).fetchone()
        conn.close()
        if row is None or not row["raw_path"]:
            return JSONResponse({"error": "Attachment not found"}, status_code=404)
        if row["mime_type"] != "application/pdf":
            return JSONResponse({"error": "render_pages only supports PDFs"}, status_code=400)
        pdf_path = Path(row["raw_path"]).resolve()
        # `is_relative_to` (Py3.9+) compares path components instead of
        # a string prefix, so `/data-evil/x.pdf` can't sneak past when
        # `data_dir == /data`.
        if not pdf_path.is_relative_to(data_dir.resolve()):
            return JSONResponse({"error": "Invalid attachment path"}, status_code=403)
        if not pdf_path.exists():
            return JSONResponse({"error": "Attachment file missing"}, status_code=404)

        try:
            import fitz  # pymupdf
        except ImportError:
            return JSONResponse({"error": "pymupdf not installed"}, status_code=500)

        def _render() -> dict:
            import base64 as _b64

            doc = fitz.open(str(pdf_path))
            try:
                total = len(doc)
                if requested_raw is None:
                    requested = list(range(1, min(total, max_pages) + 1))
                else:
                    requested = [p for p in requested_raw if 1 <= p <= total][:max_pages]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                out = []
                for page_num in requested:
                    try:
                        page = doc[page_num - 1]
                        pix = page.get_pixmap(matrix=mat)
                        png_bytes = pix.tobytes("png")
                        out.append(
                            {
                                "page": page_num,
                                "base64": _b64.b64encode(png_bytes).decode("ascii"),
                                "mime_type": "image/png",
                            }
                        )
                    except Exception as e:
                        logger.warning(f"render_pages: page {page_num} failed: {e}")
                return {"pages": out, "total_pages": total, "requested": requested}
            finally:
                doc.close()

        try:
            return await asyncio.wait_for(asyncio.to_thread(_render), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"render_pages timed out for attachment {attachment_id}")
            return JSONResponse({"error": "render timed out"}, status_code=504)
        except Exception as e:
            logger.exception(f"render_pages failed for attachment {attachment_id}: {e}")
            return JSONResponse({"error": f"render failed: {e}"}, status_code=500)

    @app.get("/api/attachment/{attachment_id}")
    async def api_attachment(
        attachment_id: int,
        user_id: str = Depends(require_user_id),
    ):
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT raw_path, mime_type, filename FROM attachments WHERE id = %s AND user_id = %s",
            (attachment_id, user_id),
        ).fetchone()
        conn.close()
        if row is None or not row["raw_path"]:
            return JSONResponse({"error": "Attachment not found"}, status_code=404)
        # Validate path is under data_dir to prevent path traversal.
        # `is_relative_to` checks path components, not string prefix, so
        # a poisoned raw_path of `/data-evil/...` can't sneak past.
        resolved = Path(row["raw_path"]).resolve()
        if not resolved.is_relative_to(data_dir.resolve()):
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
    async def api_sql_schema(user_id: str = Depends(require_user_id)):
        # Markdown description of every queryable table — surfaced to the
        # chat LLM so the sql_query tool knows the real column shapes.
        # Prepends a multi-tenant preamble so the LLM always scopes its
        # SQL to the active user — without this the model writes
        # `SELECT count(*) FROM messages` and silently sees other users'
        # rows. Phase 3g (RLS) will enforce this at the connection layer.
        from gmail_search.store.db import describe_schema_for_llm

        preamble = (
            "## Multi-tenant scoping (CRITICAL)\n\n"
            f"The active user_id is `{user_id}`. EVERY SELECT against the\n"
            "tables below MUST include `WHERE user_id = '" + user_id + "'`\n"
            "(or a join that propagates this filter). Every per-user table\n"
            "has a `user_id` column. Forgetting this filter returns rows\n"
            "from other household members, which is wrong.\n\n"
            "Examples:\n"
            "  -- correct (scopes to active user):\n"
            "  SELECT COUNT(*) FROM messages WHERE user_id = '" + user_id + "';\n"
            "  -- correct (joins keep the scope):\n"
            "  SELECT m.subject FROM messages m\n"
            "    JOIN attachments a ON a.message_id = m.id AND a.user_id = m.user_id\n"
            "   WHERE m.user_id = '" + user_id + "' AND m.id @@@ 'subject:invoice';\n\n"
            "---\n\n"
            "## Performance (write FAST SQL)\n\n"
            "BM25 search (`id @@@ 'field:term'`) is fast even across the whole\n"
            "mailbox. What's slow is **per-row text processing over `body_text`**\n"
            "(`regexp_replace`, `~*`, `substring`, `split_part`) — bodies are large\n"
            "HTML, and a broad match is tens of thousands of rows, so this blows the\n"
            "query timeout. Two rules:\n\n"
            "1. **Prefer `message_summaries.summary`** (a precomputed, cleaned, plain-\n"
            "   text summary per message) over re-deriving text from raw `body_text`.\n"
            "   Join `message_summaries ms ON ms.message_id = m.id AND ms.user_id =\n"
            "   m.user_id` and read `ms.summary` instead of regexing `m.body_text`.\n"
            "2. If you MUST touch `body_text`, **narrow first**: apply your\n"
            "   `user_id` / `date` / `@@@` filters and `LIMIT` in a CTE or subquery,\n"
            "   then run any `regexp_replace`/`~*` only on that small final set —\n"
            "   never on the full match set.\n\n"
            "---\n\n"
        )
        return {"markdown": preamble + describe_schema_for_llm()}

    @app.get("/api/status")
    async def api_status(user_id: str = Depends(require_user_id)):
        from gmail_search.store.db import JobProgress

        conn = get_connection(db_path)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages WHERE user_id = %s", (user_id,)).fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings WHERE user_id = %s", (user_id,)).fetchone()[0]
        # All dates are now UTC so string sort works correctly
        dates = conn.execute(
            "SELECT MIN(date) as oldest, MAX(date) as newest FROM messages WHERE user_id = %s",
            (user_id,),
        ).fetchone()
        total_cost = get_total_spend(conn, user_id=user_id)
        ok, spent, remaining = check_budget(conn, config["budget"]["max_usd"], user_id=user_id)
        # Search-side cost: count + sum cost from this user's `costs`
        # rows where operation='embed_query'. We can't read query_cache
        # for this — it's intentionally cross-user (same query text →
        # same vector, dedupe paid embeds across the household).
        query_stats_row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(estimated_cost_usd), 0) AS spent "
            "FROM costs WHERE operation = 'embed_query' AND user_id = %s",
            (user_id,),
        ).fetchone()
        # Backlog surface for the current prompt version. `DEFAULT_MODEL`
        # combines backend.model_id + PROMPT_VERSION, so a prompt bump
        # makes everyone's summaries count as pending and the UI shows
        # the queue draining in real time. Rate is derived from the last
        # 10 min of writes — matches what we'd compute manually.
        from gmail_search.summarize import DEFAULT_MODEL as _SUMMARY_KEY

        pending_row = conn.execute(
            """SELECT COUNT(*) FROM messages m
               LEFT JOIN message_summaries s
                 ON s.message_id = m.id AND s.user_id = m.user_id AND s.model = %s
               WHERE m.user_id = %s AND s.message_id IS NULL""",
            (_SUMMARY_KEY, user_id),
        ).fetchone()
        rate_row = conn.execute(
            """SELECT COUNT(*) FROM message_summaries
               WHERE model = %s
                 AND user_id = %s
                 AND created_at::timestamptz > NOW() - INTERVAL '10 minutes'""",
            (_SUMMARY_KEY, user_id),
        ).fetchone()
        summary_pending = int(pending_row[0] or 0)
        summary_rate_per_sec = round((int(rate_row[0] or 0)) / 600.0, 3)
        summary_eta_seconds = (
            int(summary_pending / summary_rate_per_sec) if summary_rate_per_sec > 0 and summary_pending > 0 else None
        )
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
            "query_embeds": int(query_stats_row["n"] or 0),
            "query_embed_cost_usd": round(float(query_stats_row["spent"] or 0.0), 6),
            "summary_pending": summary_pending,
            "summary_rate_per_sec": summary_rate_per_sec,
            "summary_eta_seconds": summary_eta_seconds,
            "summary_model_key": _SUMMARY_KEY,
            "running_job": running[0] if running else None,
        }

    # ── Background-job plumbing ──────────────────────────────────────────
    #
    # Long-running gmail-search subcommands (watch, update, summarize) are
    # spawned detached from the HTTP layer. Liveness is read from the
    # `job_progress` table: every daemon calls `JobProgress(...)` at
    # startup (which records `os.getpid()` + `updated_at`) and then
    # continues to bump `updated_at` either via `.update(...)` during
    # work or `.heartbeat()` when idle. A row is considered running iff
    # `status='running'` AND `updated_at` is newer than
    # `_DAEMON_STALE_SECONDS`.
    #
    # The `auth` flow still uses a pid file (`auth.pid`) because it's a
    # one-shot, not a daemon — it never writes job_progress.
    _DAEMON_STALE_SECONDS = 90

    # Map API-surface key → `job_progress.job_id` *prefix*. The actual
    # row key in multi-tenant mode is "{prefix}:{user_id}" so each user
    # gets their own job_progress row + log file, and concurrent runs
    # across users don't collide. Keeps the UI contract ("frontfill/
    # backfill/summarize") while the DB uses the CLI's subcommand names
    # ("watch/update/summarize") under per-user keys.
    _DAEMON_JOB_IDS = {
        "frontfill": "watch",
        "backfill": "update",
        "summarize": "summarize",
        "reindex": "reindex",
    }

    def _user_job_id(job_key: str, user_id: str) -> str:
        return f"{_DAEMON_JOB_IDS[job_key]}:{user_id}"

    def _user_email(user_id: str) -> str | None:
        """Look up a user's email by user_id. Used to thread --email
        into the spawned daemon so it picks the right broker entry."""
        conn = get_connection(db_path)
        try:
            row = conn.execute("SELECT email FROM users WHERE id = %s", (user_id,)).fetchone()
        finally:
            conn.close()
        return row["email"] if row else None

    def _daemon_status(job_id: str) -> dict:
        """Read the latest `job_progress` row for `job_id` and derive
        a daemon-liveness view. A daemon is considered running iff the
        row's status is `running` AND `updated_at` is fresher than the
        stale threshold. Returns:
          {running, pid, age_seconds, stage, detail}
        Missing row → {running: False, pid: None, age_seconds: None, ...}.
        """
        from datetime import datetime, timezone

        from gmail_search.store.db import JobProgress

        row = JobProgress.get(db_path, job_id)
        if not row:
            return {
                "running": False,
                "pid": None,
                "age_seconds": None,
                "stage": "",
                "detail": "",
            }
        try:
            updated = datetime.fromisoformat(row["updated_at"])
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - updated).total_seconds()
        except (ValueError, TypeError):
            age = None
        pid = row.get("pid")
        fresh = row.get("status") == "running" and (age is not None and age < _DAEMON_STALE_SECONDS)
        # A fresh heartbeat alone under-reports liveness: the backfill's
        # URL-crawl phase can wait up to 300s per blocked link, so its
        # heartbeat goes stale while the process is very much alive. Treat
        # it as running if the heartbeat is fresh OR the recorded PID is
        # still the live daemon (matched by cmdline) — else the UI shows a
        # busy daemon as "down". ("supervisor" row → "supervise" argv.)
        needle = "supervise" if job_id == "supervisor" else job_id.split(":", 1)[0]
        running = fresh or (row.get("status") == "running" and _pid_cmdline_matches(pid, needle))
        return {
            "running": bool(running),
            "pid": int(pid) if pid else None,
            "age_seconds": round(age, 1) if age is not None else None,
            "stage": row.get("stage") or "",
            "detail": row.get("detail") or "",
        }

    def _pid_cmdline_matches(pid: int | None, *needles: str) -> bool:
        """True iff /proc/<pid> exists, isn't a zombie, and its cmdline
        contains every needle. Guards os.kill against PID reuse / stale
        job_progress rows that point at a PID since recycled by an
        unrelated process running as this same user."""
        if not pid:
            return False
        try:
            with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
            return False
        return bool(cmdline) and all(n in cmdline for n in needles)

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

    def _spawn_daemon(job_key: str, log_filename: str, extra_args: list[str], user_id: str):
        """Spawn a detached daemon subprocess for a specific user.
        Threads `--email <user>` into the CLI so the subprocess pulls
        from the right broker entry, and uses a per-user job_progress
        key so two users' daemons don't collide on a single global row.
        409 if a daemon is already running for THIS user.
        """
        from fastapi.responses import JSONResponse

        from gmail_search.jobs import gmail_search_command, spawn_detached

        email = _user_email(user_id)
        if not email:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": f"no user row for {user_id}"},
            )

        job_id = _user_job_id(job_key, user_id)
        status = _daemon_status(job_id)
        if status["running"]:
            return JSONResponse(
                status_code=409,
                content={"ok": False, "error": f"already running (pid {status['pid']})"},
            )

        # Prefix the log filename with the user_id so simultaneous
        # backfills land in separate files (and cleanup is per-user).
        log_path = data_dir / f"{user_id}-{log_filename}"
        argv = gmail_search_command() + extra_args + ["--email", email, "--data-dir", str(data_dir)]
        pid = spawn_detached(argv, log_path)
        return {"ok": True, "pid": pid, "user_id": user_id, "email": email}

    def _stop_daemon(job_key: str, user_id: str):
        """SIGTERM the pid recorded in `job_progress.pid` for this
        user's daemon. Idempotent: 404 when the heartbeat says not
        running, 200 otherwise (even if the OS says the process is
        already gone).
        """
        import os
        import signal

        from fastapi.responses import JSONResponse

        job_id = _user_job_id(job_key, user_id)
        status = _daemon_status(job_id)
        if not status["running"] or not status["pid"]:
            return JSONResponse(status_code=404, content={"ok": False, "error": "not running"})
        # PID-reuse guard: confirm this exact PID is really this user's
        # daemon (subcommand + their email) before signalling it.
        subcommand = _DAEMON_JOB_IDS[job_key]
        email = _user_email(user_id) or ""
        if not _pid_cmdline_matches(status["pid"], subcommand, email):
            return JSONResponse(
                status_code=409,
                content={"ok": False, "error": "recorded pid no longer matches this daemon"},
            )
        try:
            os.kill(status["pid"], signal.SIGTERM)
            return {"ok": True, "pid": status["pid"]}
        except (ProcessLookupError, PermissionError):
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
    async def api_jobs_running(user_id: str = Depends(require_user_id)):
        """Everything /settings needs: this user's running job_progress
        rows + disk usage + per-daemon (frontfill/backfill/summarize)
        authoritative liveness derived from this user's per-key rows
        ("watch:<uid>", "update:<uid>", "summarize:<uid>"). Running
        rows are enriched with rate_per_sec + eta_seconds when the
        baseline + elapsed time permit it.
        """
        import shutil as _shutil

        from gmail_search.store.db import JobProgress

        jobs = JobProgress.get(db_path) or []
        # Show only this user's per-key rows + the legacy global rows
        # (no `:` suffix) for back-compat with daemons that haven't been
        # restarted yet. Once everything's per-user we can drop the
        # legacy filter — for now it's belt-and-braces.
        suffix = f":{user_id}"

        def _belongs(j: dict) -> bool:
            jid = j.get("job_id", "")
            return jid.endswith(suffix) or ":" not in jid

        scoped = [j for j in jobs if _belongs(j)]
        enriched = [_enrich_with_eta(j) for j in scoped]
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
            "frontfill": _daemon_status(_user_job_id("frontfill", user_id)),
            "backfill": _daemon_status(_user_job_id("backfill", user_id)),
            "summarize": _daemon_status(_user_job_id("summarize", user_id)),
        }

    # Legacy single-pool OAuth status/re-auth endpoints (token.json) were
    # removed — credentials are broker-only now. Per-user Gmail connection
    # status lives at /api/auth/gmail-status; (re)connect via
    # /api/auth/connect-gmail.

    @app.post("/api/jobs/frontfill")
    async def api_jobs_frontfill(
        interval: int = Query(120, ge=10, le=86400),
        user_id: str = Depends(require_user_id),
    ):
        """Start this user's continuous watch daemon: sync new messages
        every `interval` seconds, then extract/embed/reindex. Liveness
        is tracked via the `watch:<user_id>` row in `job_progress`.
        """
        return _spawn_daemon(
            job_key="frontfill",
            log_filename="watch.log",
            extra_args=["watch", "--interval", str(interval)],
            user_id=user_id,
        )

    @app.post("/api/jobs/frontfill/stop")
    async def api_jobs_frontfill_stop(user_id: str = Depends(require_user_id)):
        return _stop_daemon("frontfill", user_id)

    @app.post("/api/jobs/backfill")
    async def api_jobs_backfill(
        min_free_gb: float = Query(5.0, ge=0.1, le=10000.0),
        user_id: str = Depends(require_user_id),
    ):
        """Pull this user's older messages through the full pipeline.
        Runs forever via `--loop`: once caught up (or if a cycle
        crashes), sleeps 5 min and resumes. Pauses batches when free
        disk < min_free_gb.
        """
        return _spawn_daemon(
            job_key="backfill",
            log_filename="backfill.log",
            extra_args=["update", "--min-free-gb", str(min_free_gb), "--loop"],
            user_id=user_id,
        )

    @app.post("/api/jobs/backfill/stop")
    async def api_jobs_backfill_stop(user_id: str = Depends(require_user_id)):
        return _stop_daemon("backfill", user_id)

    @app.post("/api/jobs/summarize")
    async def api_jobs_summarize(
        concurrency: int = Query(12, ge=1, le=64),
        batch_size: int = Query(1, ge=1, le=16),
        user_id: str = Depends(require_user_id),
    ):
        """Run `gmail-search summarize` for this user in the background.
        Rate + ETA land in /api/jobs/running via the
        `summarize:<user_id>` row.
        """
        return _spawn_daemon(
            job_key="summarize",
            log_filename="summarize.log",
            extra_args=[
                "summarize",
                "--concurrency",
                str(concurrency),
                "--batch-size",
                str(batch_size),
            ],
            user_id=user_id,
        )

    @app.post("/api/jobs/summarize/stop")
    async def api_jobs_summarize_stop(user_id: str = Depends(require_user_id)):
        return _stop_daemon("summarize", user_id)

    # ── Admin: per-user sync management ──────────────────────────────
    # Admins (GMS_ADMIN_EMAILS, default scottmsilver@gmail.com) get a
    # cross-user view + can start/stop other users' daemons. The
    # supervisor handles auto-start in the steady state — these
    # endpoints are for forced reload (re-pull a user's mail), pausing
    # an account during a trip, or debugging a stuck daemon.
    from gmail_search.auth import require_admin

    # Cache broker Gmail-connection probes so the /admin poll (every few
    # seconds) doesn't hit the broker /token endpoint once per user per
    # poll. Keyed by email → (fetched_at, status).
    _gmail_status_cache: dict[str, tuple[float, dict]] = {}
    _GMAIL_STATUS_TTL = 60.0

    def _user_gmail_status(email: str) -> dict:
        """Best-effort broker Gmail-connection status for `email`:
        `{connected: bool, problem: 'scope'|None}`. Cached for
        `_GMAIL_STATUS_TTL`s. Never raises — a broker hiccup or a
        revoked/absent token just reports `connected=False` so the UI
        can prompt a (re)connect."""
        import time as _time

        from gmail_search.gmail.auth import _broker_credentials_for

        now = _time.time()
        hit = _gmail_status_cache.get(email)
        if hit and (now - hit[0]) < _GMAIL_STATUS_TTL:
            return hit[1]
        status = {"connected": False, "problem": None}
        try:
            creds = _broker_credentials_for(email)
            status["connected"] = bool(creds and creds.token)
        except PermissionError:
            status["problem"] = "scope"
        except Exception:
            pass
        _gmail_status_cache[email] = (now, status)
        return status

    @app.get("/api/admin/users")
    async def api_admin_users(_admin=Depends(require_admin)):
        """Return every user + their per-user daemon health (frontfill,
        backfill, summarize) + sync_enabled flag. Powers the /admin UI."""
        conn = get_connection(db_path)
        try:
            rows = conn.execute(
                """SELECT u.id, u.email, u.name, u.sync_enabled,
                          u.invited_at, u.last_login_at,
                          (SELECT COUNT(*) FROM messages m WHERE m.user_id = u.id) AS msg_count,
                          (SELECT COUNT(*) FROM embeddings e WHERE e.user_id = u.id) AS emb_count
                   FROM users u
                   ORDER BY u.email"""
            ).fetchall()
            from gmail_search.gmail.auth import get_credential_health

            health_by_uid = {r["id"]: get_credential_health(conn, r["id"]) for r in rows}
        finally:
            conn.close()
        users = []
        for r in rows:
            uid = r["id"]
            users.append(
                {
                    "id": uid,
                    "email": r["email"],
                    "name": r["name"],
                    "sync_enabled": bool(r["sync_enabled"]),
                    "invited_at": r["invited_at"].isoformat() if r["invited_at"] else None,
                    "last_login_at": r["last_login_at"].isoformat() if r["last_login_at"] else None,
                    "msg_count": int(r["msg_count"] or 0),
                    "emb_count": int(r["emb_count"] or 0),
                    "frontfill": _daemon_status(_user_job_id("frontfill", uid)),
                    "backfill": _daemon_status(_user_job_id("backfill", uid)),
                    "summarize": _daemon_status(_user_job_id("summarize", uid)),
                    "reindex": _daemon_status(_user_job_id("reindex", uid)),
                    "gmail": _user_gmail_status(r["email"]),
                    "credential_health": health_by_uid.get(uid),
                }
            )
        # Plus the global supervisor row so admin can see if it's alive.
        return {"users": users, "supervisor": _daemon_status("supervisor")}

    @app.get("/api/admin/progress")
    async def api_admin_progress(_admin=Depends(require_admin)):
        """Work-remaining stats for /admin: per-user embedding coverage
        (messages embedded vs total) and the URL-crawl queue split into
        fast lane (never tried), slow lane (failed, backing off), abandoned
        (hit the retry cap), and done. This is the heavier query (a scan of
        the URL-stub attachments), so the UI polls it on a slower interval
        than the daemon-health poll."""
        from gmail_search.store.queries import _MAX_CRAWL_ATTEMPTS

        conn = get_connection(db_path)
        try:
            emb_rows = conn.execute(
                """SELECT u.id, u.email,
                          (SELECT COUNT(*) FROM messages m WHERE m.user_id = u.id) AS messages,
                          (SELECT COUNT(DISTINCT e.message_id) FROM embeddings e
                             WHERE e.user_id = u.id AND e.chunk_type = 'message') AS embedded
                   FROM users u
                  ORDER BY u.email"""
            ).fetchall()
            crawl = conn.execute(
                """SELECT
                     COUNT(*) FILTER (WHERE extracted_text IS NULL AND crawl_attempts = 0) AS fast,
                     COUNT(*) FILTER (WHERE extracted_text IS NULL AND crawl_attempts BETWEEN 1 AND %s) AS slow,
                     COUNT(*) FILTER (WHERE extracted_text IS NULL AND crawl_attempts >= %s) AS dead,
                     COUNT(*) FILTER (WHERE extracted_text IS NOT NULL) AS done
                   FROM attachments
                  WHERE mime_type = 'text/html' AND filename LIKE %s""",
                (_MAX_CRAWL_ATTEMPTS - 1, _MAX_CRAWL_ATTEMPTS, "URL: %"),
            ).fetchone()
        finally:
            conn.close()

        embedding = []
        for r in emb_rows:
            msgs = int(r["messages"] or 0)
            emb = int(r["embedded"] or 0)
            embedding.append(
                {
                    "id": r["id"],
                    "email": r["email"],
                    "messages": msgs,
                    "embedded": emb,
                    "pending": max(0, msgs - emb),
                    "pct": round(100.0 * emb / msgs, 1) if msgs else 100.0,
                }
            )
        c = crawl or {}
        fast, slow, dead, done = (int(c[k] or 0) for k in ("fast", "slow", "dead", "done"))
        return {
            "embedding": embedding,
            "crawl": {
                "fast": fast,
                "slow": slow,
                "dead": dead,
                "done": done,
                "pending": fast + slow,
                "max_attempts": _MAX_CRAWL_ATTEMPTS,
            },
        }

    @app.post("/api/admin/users/{user_id}/sync_enabled")
    async def api_admin_set_sync_enabled(
        user_id: str,
        payload: dict = Body(...),
        _admin=Depends(require_admin),
    ):
        """Toggle the supervisor's interest in a user. `enabled=false`
        means the supervisor stops auto-spawning their daemons next
        cycle (existing daemons keep running until they exit
        naturally; admin can stop them explicitly via the per-job
        endpoints below). `enabled=true` re-enrolls them."""
        enabled = bool(payload.get("enabled"))
        conn = get_connection(db_path)
        try:
            cur = conn.execute(
                "UPDATE users SET sync_enabled = %s WHERE id = %s RETURNING email",
                (enabled, user_id),
            )
            row = cur.fetchone()
            conn.commit()
        finally:
            conn.close()
        if row is None:
            return JSONResponse({"ok": False, "error": "no such user"}, status_code=404)
        return {"ok": True, "user_id": user_id, "email": row["email"], "sync_enabled": enabled}

    @app.post("/api/admin/users/{user_id}/{job_key}/start")
    async def api_admin_start_user_daemon(
        user_id: str,
        job_key: str,
        _admin=Depends(require_admin),
    ):
        """Force-start a user's daemon. Equivalent to clicking
        Frontfill/Backfill/Summarize in their Settings — but admin
        can do it for any user. Returns 400 for unknown job_key."""
        if job_key not in _DAEMON_JOB_IDS:
            return JSONResponse(
                {"ok": False, "error": f"job_key must be one of {sorted(_DAEMON_JOB_IDS)}"},
                status_code=400,
            )
        log_filename = {
            "frontfill": "watch.log",
            "backfill": "backfill.log",
            "summarize": "summarize.log",
            "reindex": "reindex.log",
        }[job_key]
        extra_args = {
            "frontfill": ["watch", "--interval", "120"],
            "backfill": ["update", "--loop"],
            "summarize": ["summarize", "--concurrency", "12", "--batch-size", "1", "--loop"],
            "reindex": ["reindex", "--loop", "--quantum", "60", "--min-new", "10000", "--max-age", "900"],
        }[job_key]
        return _spawn_daemon(
            job_key=job_key,
            log_filename=log_filename,
            extra_args=extra_args,
            user_id=user_id,
        )

    @app.post("/api/admin/users/{user_id}/{job_key}/stop")
    async def api_admin_stop_user_daemon(
        user_id: str,
        job_key: str,
        _admin=Depends(require_admin),
    ):
        if job_key not in _DAEMON_JOB_IDS:
            return JSONResponse(
                {"ok": False, "error": f"job_key must be one of {sorted(_DAEMON_JOB_IDS)}"},
                status_code=400,
            )
        return _stop_daemon(job_key, user_id)

    @app.post("/api/admin/supervisor/start")
    async def api_admin_supervisor_start(_admin=Depends(require_admin)):
        """Start the multi-user supervisor — the reconciler that keeps
        each sync-enabled user's watch/update/summarize alive and
        converges disabled users down. 409 if already running. Spawned
        as a child of this server process so it inherits the broker env
        (and therefore so do the daemons it spawns)."""
        if _daemon_status("supervisor")["running"]:
            return JSONResponse({"ok": False, "error": "supervisor already running"}, status_code=409)
        from gmail_search.jobs import gmail_search_command, spawn_detached

        argv = gmail_search_command() + ["supervise", "--interval", "30", "--data-dir", str(data_dir)]
        pid = spawn_detached(argv, data_dir / "supervise.log")
        return {"ok": True, "pid": pid}

    @app.post("/api/admin/supervisor/stop")
    async def api_admin_supervisor_stop(_admin=Depends(require_admin)):
        """SIGTERM the supervisor. Supervised daemons keep running — the
        supervisor only respawns/converges, it doesn't own them."""
        import os as _os
        import signal as _signal

        from gmail_search.store.db import JobProgress

        row = JobProgress.get(db_path, "supervisor")
        pid = row.get("pid") if row else None
        # PID-reuse guard: only signal a PID whose cmdline is actually the
        # supervisor (a stale row could point at a recycled PID).
        if not pid or not _pid_cmdline_matches(pid, "supervise"):
            return JSONResponse({"ok": False, "error": "supervisor not running"}, status_code=404)
        try:
            _os.kill(int(pid), _signal.SIGTERM)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        return {"ok": True, "pid": pid}

    return app
