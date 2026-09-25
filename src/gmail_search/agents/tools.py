"""Retrieval tools over the existing HTTP surface, served to deep-mode
runtimes by `mcp_tools_server.py`.

They reuse our live `/api/search`, `/api/query`,
`/api/thread/<id>`, `/api/sql` endpoints rather than calling Python
helpers directly. Two wins:
  1. Same security gate as the chat agent sees — `/api/sql`'s
     read-only + forbidden-keyword check runs on every query, no
     matter which client made it.
  2. One pattern both for Python deep-mode and TypeScript chat-mode
     paths; bugs discovered in one get fixed in the other.

All tool functions return JSON-safe dicts (not pydantic models),
which is what the MCP server hands back to the model as tool results. Strings are clipped where it matters so a chatty endpoint
(long SQL result sets, huge thread bodies) can't blow the model's
context window on a single call.

Tools are bound to a base URL so tests can redirect at a different
port if needed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)


# Match the caps the TS-side tools use so both sides' Writer sees
# similarly-shaped evidence. If this drifts, retrieval quality
# between chat mode and deep mode diverges.
SEARCH_ENRICH_BODY_CHARS = 4000
QUERY_ENRICH_BODY_CHARS = 2000
THREAD_BODY_CHAR_CAP = 20_000
SQL_CELL_CHAR_CAP = 8_000


# HTTP timeout for the in-process /api/* calls. Env-overridable so it can be
# tuned without a code change (never a hardcoded URL — just a duration). A
# single search is normally <2s; the headroom covers concurrent batch fan-out.
def _agent_http_timeout() -> float:
    """Parse GMAIL_AGENT_HTTP_TIMEOUT, falling back to 60s on a bad/empty/
    non-positive value. Must never raise: this runs at import time and a
    crash here would take down both serve and the MCP tools server."""
    try:
        v = float(os.environ.get("GMAIL_AGENT_HTTP_TIMEOUT", "60"))
        return v if v > 0 else 60.0
    except (TypeError, ValueError):
        return 60.0


DEFAULT_TIMEOUT_SECONDS = _agent_http_timeout()


def _default_base_url() -> str:
    """Our Python HTTP surface. Defaults to the serve command's local
    bind; env override is kept because the tools can run in a
    container-per-request future where the host/port differ."""
    return os.environ.get("GMAIL_SEARCH_API_URL", "http://127.0.0.1:8090").rstrip("/")


def _clip(text: str, cap: int) -> str:
    """Truncate with an explicit "truncated: N chars" marker so the
    downstream model knows content was cut. Unchanged content passes
    through unaltered."""
    if text is None:
        return ""
    if len(text) <= cap:
        return text
    return text[: cap - 32] + f"… [truncated: original {len(text)} chars]"


def _service_headers(user_id: str | None) -> dict[str, str]:
    """Build the headers MCP/agent tools attach to /api/* calls so
    `require_user_id` on the FastAPI side scopes the request to the
    deep-mode session's user. The admin-token check is constant-time
    on the receiving end; without the token, X-User-Id is ignored
    (raises 401), so the bearer + X-User-Id pair is what unlocks the
    cross-user scoping path.

    Also propagates the current trace id outbound as a W3C `traceparent`
    header (when a trace is active), so a tool call's serve request + DB query
    join to the originating MCP call on one id."""
    from gmail_search.trace import current_trace_id, make_traceparent

    headers: dict[str, str] = {}
    tid = current_trace_id()
    if tid:
        headers["traceparent"] = make_traceparent(tid)
    if user_id is None:
        return headers
    admin = os.environ.get("GMAIL_MCP_ADMIN_TOKEN")
    headers["X-User-Id"] = user_id
    if admin:
        # Without the token, X-User-Id is ignored on the serve side (401),
        # so it falls back to the bootstrap user — matches single-pool behaviour.
        headers["Authorization"] = f"Bearer {admin}"
    return headers


async def _get(
    path: str,
    *,
    params: dict | None = None,
    base_url: str | None = None,
    user_id: str | None = None,
) -> dict:
    """ASYNC httpx call. Critical: the retrieval tools run INSIDE the
    same FastAPI process that serves /api/search + friends. A sync
    client call from an async request handler deadlocks the event
    loop (tool waits on the socket, uvicorn can't accept the new
    inbound request because its event loop is blocked). Async
    yields while waiting, so the inbound handler runs and replies.

    `user_id`, when given, scopes the call to that user via the
    service-token + X-User-Id pair. None => single-pool fallback.

    4xx / 5xx responses are NOT raised — they're returned as
    `{error: "<body>"}` so the LLM sees the server's message as a
    tool result and can retry with a corrected query. Raising would
    crash the whole turn on one bad SQL syntax mistake.
    """
    url = (base_url or _default_base_url()) + path
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            resp = await client.get(url, params=params or {}, headers=_service_headers(user_id))
            if resp.status_code >= 400:
                return _error_payload(resp)
            return resp.json()
    except httpx.HTTPError as exc:
        # Transport-level failures (timeouts, connection resets) must NOT raise:
        # a single slow search would otherwise propagate through the batch
        # gather and kill every other search in the call. Surface it as a
        # tool-visible error instead, like 4xx/5xx above.
        return {"error": f"request failed: {type(exc).__name__}: {exc}", "kind": "transport"}


async def _post(
    path: str,
    *,
    json: dict,
    base_url: str | None = None,
    user_id: str | None = None,
) -> dict:
    url = (base_url or _default_base_url()) + path
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=json, headers=_service_headers(user_id))
            if resp.status_code >= 400:
                return _error_payload(resp)
            return resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"request failed: {type(exc).__name__}: {exc}", "kind": "transport"}


def _error_payload(resp) -> dict:
    """Turn an upstream error response into a tool result the model
    can reason about. Tries to extract the server's JSON error
    field; falls back to the raw body."""
    try:
        body = resp.json()
        if isinstance(body, dict) and "error" in body:
            return {"error": str(body["error"]), "status": resp.status_code}
    except Exception:  # noqa: BLE001
        pass
    return {"error": resp.text[:500], "status": resp.status_code}


# ── search_emails ──────────────────────────────────────────────────


async def search_emails(
    query: str,
    date_from: str = "",
    date_to: str = "",
    top_k: int = 10,
    detail: str = "snippet",
    max_matches: int = 3,
    *,
    user_id: str | None = None,
) -> dict:
    """Search for messages by relevance (semantic + BM25 blend).

    Use this for "what did we decide about X" / "find messages
    mentioning Y" questions. Empty `date_from` / `date_to` skip the
    date filter. `top_k` caps the number of threads returned; default
    10 is a sensible middle ground.

    `detail` controls how much content comes back per match — pick the
    cheapest level that answers your question, since bigger levels cost
    far more tokens:
      - "refs": ONE LINE per thread — {thread_id, subject, date_last,
        from, score}, no matches array. Use for fan-out inventory
        questions ("which of these N things has an email?") where you
        only need to know a thread exists and where to look next.
      - "snippet" (default): the matched-text snippet only. Best for
        inventories / "how many" / locating which threads to look at.
      - "summary": + a one-line LLM summary of each matched message.
        Often enough to answer without opening the thread.
      - "full": + the WHOLE email body of each match. Use when you
        need to actually read the matches; avoids N get_thread calls
        but can be large, so keep top_k modest.

    `max_matches` caps how many matching messages come back per thread
    (top-scoring first; default 3, 0 = unlimited). When the cap bites,
    the thread carries `matches_truncated` with the dropped count —
    re-query with a higher cap if you need the rest.

    Returns {"results": [{thread_id, cite_ref, subject, participants,
    score, matches: [...]}]}. Each match always carries the top-hit
    message id + snippet; "summary"/"full" add `summary`/`body`.
    """
    params: dict[str, Any] = {
        "q": query,
        "k": top_k,
        "match_detail": detail,
        "max_matches": max_matches,
        # Facets exist for the web UI's filter chips; agents never read
        # them, so the agent path always opts out.
        "include_facets": "false",
    }
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    data = await _get("/api/search", params=params, user_id=user_id)
    # cite_ref = full thread_id. Earlier we used the first 8 hex chars
    # as a shorter shorthand, but that's a 32-bit namespace and
    # collisions occur in real mailboxes (e.g. two emails arriving in
    # the same second from the same sender). The UI shortens chips
    # visually anyway, so there's no benefit to truncating here.
    for row in data.get("results", []):
        row.setdefault("cite_ref", row.get("thread_id") or "")
    return data


# ── find_facts ─────────────────────────────────────────────────────


async def find_facts(
    query: str,
    *,
    exhaustive: bool = True,
    k: int = 200,
    user_id: str | None = None,
) -> dict:
    """ENUMERATE every instance of an entity/attribute across the whole
    mailbox in ONE call — instead of issuing many `search_emails`
    reformulations and hoping you covered them all.

    Use this for exhaustive "list ALL my X" questions: all my license
    plates, all my account numbers, every flight I've booked, every
    address I've lived at. It runs hybrid (semantic ∪ keyword) retrieval
    over pre-extracted atomic facts ("propositions") mined from the
    mailbox, so bare identifiers (plates, VINs, codes) that embeddings
    rank near-zero are still recalled via the BM25 half.

    Returns `{"facts": [{fact, message_id, thread_id, cosine, bm25}, ...]}`.
    Each fact carries a `message_id` back-pointer — use it (via
    `get_thread`) to cite and verify the source before reporting it.
    If propositions haven't been backfilled yet, `facts` is `[]`.

    `exhaustive=True` (default) returns the full candidate set above the
    similarity floor; `k` caps the number of facts returned.
    """
    params: dict[str, Any] = {
        "q": query,
        "exhaustive": str(exhaustive).lower(),
        "k": k,
        "hybrid": "true",
    }
    return await _get("/api/find_facts", params=params, user_id=user_id)


# ── query_emails ───────────────────────────────────────────────────


async def query_emails(
    sender: str = "",
    subject_contains: str = "",
    date_from: str = "",
    date_to: str = "",
    label: str = "",
    has_attachment: bool | None = None,
    order_by: str = "date_desc",
    limit: int = 20,
    *,
    user_id: str | None = None,
) -> dict:
    """Filter messages by structured metadata. Use when you know WHAT
    fields to filter on — no relevance ranking. Returns a list of
    threads in the requested order (`date_desc` default).

    Parameters:
      sender: substring match on From: (e.g. `@dartmouth.edu`)
      subject_contains: substring match on Subject:
      date_from / date_to: ISO dates `YYYY-MM-DD` (inclusive)
      label: Gmail label to filter on (e.g. `INBOX`, `IMPORTANT`)
      has_attachment: true/false to filter on presence of attachments
      order_by: `date_desc` or `date_asc`
      limit: max threads returned
    """
    params: dict[str, Any] = {"order_by": order_by, "limit": limit}
    if sender:
        params["sender"] = sender
    if subject_contains:
        params["subject_contains"] = subject_contains
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if label:
        params["label"] = label
    if has_attachment is not None:
        params["has_attachment"] = str(has_attachment).lower()
    data = await _get("/api/query", params=params, user_id=user_id)
    for row in data.get("results", []):
        row.setdefault("cite_ref", row.get("thread_id") or "")
    return data


# ── get_thread ─────────────────────────────────────────────────────


def _read_options_error(body_format, message_ids, body_offset, body_limit):
    if body_format not in ("markdown", "raw"):
        return "body_format must be markdown or raw"
    if type(body_offset) is not int or body_offset < 0:
        return "body_offset must be a non-negative integer"
    if type(body_limit) is not int or not 1 <= body_limit <= 100_000:
        return "body_limit must be between 1 and 100000 characters"
    if message_ids is not None and (not isinstance(message_ids, list) or not message_ids or
                                   len(message_ids) > 1000 or not all(isinstance(x, str) and x for x in message_ids)):
        return "message_ids must be a non-empty list of at most 1000 message IDs, or omitted"
    return None


def _body_page(message, field, value, offset, limit):
    message[field] = value[offset:offset + limit]
    prefix = "body" if field == "body_text" else field
    message[f"{prefix}_total_chars"] = len(value)
    message[f"{prefix}_next_offset"] = offset + limit if offset + limit < len(value) else None
    if offset or len(value) > limit:
        message[f"{field}_truncated"] = True
        if field == "body_text":
            message["original_chars"] = len(value)


async def get_thread(
    thread_id: str, *, user_id: str | None = None,
    body_format: Literal["markdown", "raw"] = "markdown",
    message_ids: list[str] | None = None, body_offset: int = 0,
    body_limit: int = THREAD_BODY_CHAR_CAP,
) -> dict:
    """Read selected messages as Markdown, or explicitly request original bodies.

    The default page is 20k characters per message. Continuation offsets refer
    to the chosen representation; original_chars is the full body length.
    Stored content and the web thread response are never changed.
    """
    from gmail_search.agents.mail_content import readable_body

    error = _read_options_error(body_format, message_ids, body_offset, body_limit)
    if error:
        return {"error": error}
    data = await _get(f"/api/thread/{thread_id}", user_id=user_id)
    if "error" in data:
        return data
    cite_ref = data.get("thread_id") or thread_id
    originals = data.get("messages", [])
    selected = set(message_ids) if message_ids is not None else None
    messages = []
    for original in originals:
        if selected is not None and original.get("id") not in selected:
            continue
        msg = dict(original)
        msg.update(cite_ref=cite_ref, body_offset=body_offset, body_limit=body_limit)
        if body_format == "raw":
            msg["body_format"] = "raw"
            _body_page(msg, "body_text", original.get("body_text") or "", body_offset, body_limit)
            _body_page(msg, "body_html", original.get("body_html") or "", body_offset, body_limit)
        else:
            body, source = readable_body(original.get("body_text") or "", original.get("body_html") or "")
            msg.pop("body_html", None)
            msg["body_source"] = source
            msg["body_format"] = "markdown" if source == "html" else "text"
            _body_page(msg, "body_text", body, body_offset, body_limit)
        messages.append(msg)
    return {**data, "cite_ref": cite_ref, "thread_message_count": len(originals),
            "returned_message_count": len(messages), "messages": messages}


# Cap on how many items one batch tool call can request. Started at
# 20; first session with batch tools immediately filled multiple
# `get_thread_batch` calls to the 20-cap, so the agent clearly wants
# to go bigger. Bumped to 100. Postgres handles 100 concurrent
# read-only queries fine; the cap mainly exists to bound a runaway
# plan from queueing thousands of concurrent requests.
BATCH_MAX_ITEMS = 100


async def _gather_batch(make_coro, items: list) -> list:
    """Run one coroutine per item concurrently, isolating per-item failures.

    Any item that raises — whether building the coroutine (e.g. `**item` with
    a bad key) or running it (an exception escaping the underlying tool) —
    becomes an `{"error": ...}` result instead of aborting the whole batch.
    This is what makes the *_batch tools' documented per-item error contract
    actually hold: previously a bare `gather` let one failure (e.g. an httpx
    timeout on a single slow search) propagate and kill every sibling.
    """
    import asyncio as _asyncio

    async def _one(item):
        try:
            coro = make_coro(item)
        except Exception as exc:  # noqa: BLE001 — malformed item (bad/missing keys)
            return {"error": f"invalid item: {type(exc).__name__}: {exc}"}
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001 — exception escaping the tool
            return {"error": f"{type(exc).__name__}: {exc}"}

    return await _asyncio.gather(*[_one(it) for it in items])


async def search_emails_batch(searches: list[dict], *, user_id: str | None = None) -> dict:
    """Run many semantic searches concurrently. Each item is a dict
    matching `search_emails`'s signature: `{query, date_from?,
    date_to?, top_k?, detail?, max_matches?}`. `detail` is "refs" (one
    line per thread), "snippet" (default, compact), "summary", or
    "full" (whole email body per match) — pick the cheapest level that
    answers the question. For fan-out inventory questions across many
    queries, use `detail="refs"`. Use this whenever you have ≥1 search
    to issue — even one search goes through the batch tool.

    Returns `{"results": [{"input": <input dict>, "result": <search_emails shape>}, ...]}`.
    Per-search errors land in that entry's `result` as `{"error": ...}`.

    Cap: BATCH_MAX_ITEMS searches per call.
    """
    if not isinstance(searches, list) or not searches:
        return {"error": "searches must be a non-empty list of dicts"}
    if len(searches) > BATCH_MAX_ITEMS:
        return {"error": f"searches cap is {BATCH_MAX_ITEMS}; got {len(searches)}. Split into multiple batches."}
    results = await _gather_batch(lambda s: search_emails(**s, user_id=user_id), searches)
    return {"results": [{"input": s, "result": r} for s, r in zip(searches, results)]}


async def query_emails_batch(filters: list[dict], *, user_id: str | None = None) -> dict:
    """Run many structured filters concurrently. Each item is a dict
    matching `query_emails`'s signature: any of `{sender,
    subject_contains, date_from, date_to, label, has_attachment,
    order_by, limit}`.

    Returns `{"results": [{"input": <input dict>, "result": <query_emails shape>}, ...]}`.

    Cap: BATCH_MAX_ITEMS filters per call.
    """
    if not isinstance(filters, list) or not filters:
        return {"error": "filters must be a non-empty list of dicts"}
    if len(filters) > BATCH_MAX_ITEMS:
        return {"error": f"filters cap is {BATCH_MAX_ITEMS}; got {len(filters)}. Split into multiple batches."}
    results = await _gather_batch(lambda f: query_emails(**f, user_id=user_id), filters)
    return {"results": [{"input": f, "result": r} for f, r in zip(filters, results)]}


async def get_thread_batch(
    thread_ids: list[str], *, user_id: str | None = None,
    body_format: Literal["markdown", "raw"] = "markdown",
    message_ids: list[str] | None = None, body_offset: int = 0,
    body_limit: int = THREAD_BODY_CHAR_CAP,
) -> dict:
    """Fetch many threads concurrently in a single tool call. Use
    this — not N sequential `get_thread` calls — whenever you need
    multiple threads' bodies. Same per-thread response shape as
    `get_thread`; ordering matches the input.

    Returns `{"results": [{"thread_id": "...", "result": <get_thread shape>}, ...]}`.
    Per-thread errors land in that thread's `result` as `{"error": ...}` —
    the batch as a whole still succeeds so the agent gets every other
    thread's data without retrying the whole thing.

    Cap: BATCH_MAX_ITEMS thread_ids per call. Beyond that, split.
    """
    if not isinstance(thread_ids, list) or not thread_ids:
        return {"error": "thread_ids must be a non-empty list of strings"}
    if len(thread_ids) > BATCH_MAX_ITEMS:
        return {"error": f"thread_ids cap is {BATCH_MAX_ITEMS}; got {len(thread_ids)}. Split into multiple batches."}
    error = _read_options_error(body_format, message_ids, body_offset, body_limit)
    if error:
        return {"error": error}
    results = await _gather_batch(lambda tid: get_thread(
        tid, user_id=user_id, body_format=body_format, message_ids=message_ids,
        body_offset=body_offset, body_limit=body_limit,
    ), thread_ids)
    return {"results": [{"thread_id": tid, "result": r} for tid, r in zip(thread_ids, results)]}


# ── sql_query ──────────────────────────────────────────────────────


async def sql_query(query: str, *, user_id: str | None = None) -> dict:
    """Return a stable denial for callers retaining a stale SQL tool handle."""
    from gmail_search.sql_access import raw_sql_disabled

    return raw_sql_disabled()


async def sql_query_batch(queries: list[str], *, user_id: str | None = None) -> dict:
    """Run many independent SELECT queries concurrently in a single
    tool call. Use this — not N sequential `sql_query` calls — when
    you have multiple independent queries (different time windows,
    senders, year-buckets, ticket numbers).

    Returns `{"results": [{"query": "...", "result": <sql_query shape>}, ...]}`.
    Per-query errors (validation rejection, runtime error) land in
    that query's `result` as `{"error": ...}`; the batch as a whole
    still succeeds so the agent gets every other query's data
    without retrying the whole batch.

    Each query goes through the same safety gate as `sql_query`
    (BM25 enforcement, read-only, 500-row + 10s timeout each).
    Cap: BATCH_MAX_ITEMS queries per call. Beyond that, split.
    """
    if not isinstance(queries, list) or not queries:
        return {"error": "queries must be a non-empty list of SQL strings"}
    if len(queries) > BATCH_MAX_ITEMS:
        return {"error": f"queries cap is {BATCH_MAX_ITEMS}; got {len(queries)}. Split into multiple batches."}
    results = await _gather_batch(lambda q: sql_query(q, user_id=user_id), queries)
    return {"results": [{"query": q, "result": r} for q, r in zip(queries, results)]}


# ── describe_schema ────────────────────────────────────────────────


async def describe_schema(*, user_id: str | None = None) -> dict:
    """Return the markdown schema documentation for every table the
    `sql_query` tool can read. Use this BEFORE writing a non-trivial
    `sql_query` if you're unsure about a column name — the docs include
    canonical names (e.g. `from_addr` not `sender`, `body_text` not
    `body`, `id` not `message_id`), per-column notes, and BM25
    guidance. Cheap call; result is short (~few KB).

    Multi-tenant: forwards the session's user_id so the response
    includes a preamble pinning the active user_id — the LLM uses
    this to scope every SELECT it writes."""
    return await _get("/api/sql_schema", user_id=user_id)


# ── get_attachment ─────────────────────────────────────────────────


_ATTACHMENT_VALID_MODES = ("meta", "text", "rendered_pages", "raw")


async def get_attachment(
    attachment_id: int, mode: str = "text", inline: bool = True, *, user_id: str | None = None
) -> dict:
    """Fetch an email attachment in one of four shapes:

    - `meta`: filename + mime_type + size_bytes + thread_id, no body.
      Cheap, useful for "what's attached?" queries before deciding
      whether to read.
    - `text`: extracted text (PDF / docx / OCR'd images). Run this
      when you want the attachment's words.
    - `rendered_pages`: PDF pages rasterized to base64-PNG. Heavy
      (multiple MB per page), token-expensive — only use when text
      extraction is empty or unhelpful (scans, complex layouts) AND
      the model can read images.
    - `raw`: the original bytes. Returns metadata + sha256 + base64 of
      the bytes (default, for files <= ~1 MB) AND a signed `fetch_url`
      (via the MCP host, expires ~15 min) you can GET directly with no
      auth — use that for files too big to inline. Pass `inline=false`
      for url-only.

    `attachment_id` comes from `get_thread`'s `attachments[*].id`."""
    if mode not in _ATTACHMENT_VALID_MODES:
        return {"error": f"invalid mode {mode!r}; must be one of {list(_ATTACHMENT_VALID_MODES)}"}
    if mode == "meta":
        return await _get(f"/api/attachment/{attachment_id}/meta", user_id=user_id)
    if mode == "text":
        return await _get(f"/api/attachment/{attachment_id}/text", user_id=user_id)
    if mode == "raw":
        return await _get(
            f"/api/attachment/{attachment_id}/raw",
            params={"inline": "true" if inline else "false"},
            user_id=user_id,
        )
    return await _get(f"/api/attachment/{attachment_id}/render_pages", user_id=user_id)


async def get_attachment_batch(items: list[dict], *, user_id: str | None = None) -> dict:
    """Fetch many attachments concurrently. Each item is a dict
    matching `get_attachment`'s signature: `{attachment_id, mode?}`
    where `mode` defaults to `"text"`.

    Returns `{"results": [{"input": <input dict>, "result": <get_attachment shape>}, ...]}`.

    Cap: BATCH_MAX_ITEMS attachments per call.
    """
    if not isinstance(items, list) or not items:
        return {"error": "items must be a non-empty list of dicts"}
    if len(items) > BATCH_MAX_ITEMS:
        return {"error": f"items cap is {BATCH_MAX_ITEMS}; got {len(items)}. Split into multiple batches."}
    results = await _gather_batch(lambda it: get_attachment(**it, user_id=user_id), items)
    return {"results": [{"input": it, "result": r} for it, r in zip(items, results)]}
