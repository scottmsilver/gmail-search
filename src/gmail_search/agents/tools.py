"""ADK FunctionTool wrappers for the existing HTTP retrieval surface.

The Retriever agent reuses our live `/api/search`, `/api/query`,
`/api/thread/<id>`, `/api/sql` endpoints rather than calling Python
helpers directly. Two wins:
  1. Same security gate as the chat agent sees — `/api/sql`'s
     read-only + forbidden-keyword check runs on every query, no
     matter which client made it.
  2. One pattern both for Python deep-mode and TypeScript chat-mode
     paths; bugs discovered in one get fixed in the other.

All tool functions return JSON-safe dicts (not pydantic models)
because that's what ADK's LlmAgent feeds back to the model as tool
results. Strings are clipped where it matters so a chatty endpoint
(long SQL result sets, huge thread bodies) can't blow the model's
context window on a single call.

Tools are bound to a base URL so tests can redirect at a different
port if needed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# Match the caps the TS-side tools use so both sides' Writer sees
# similarly-shaped evidence. If this drifts, retrieval quality
# between chat mode and deep mode diverges.
SEARCH_ENRICH_BODY_CHARS = 4000
QUERY_ENRICH_BODY_CHARS = 2000
THREAD_BODY_CHAR_CAP = 20_000
SQL_CELL_CHAR_CAP = 8_000
DEFAULT_TIMEOUT_SECONDS = 30


def _default_base_url() -> str:
    """Our Python HTTP surface. Defaults to the serve command's local
    bind; env override is kept because the ADK tools can run in a
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


async def _get(path: str, *, params: dict | None = None, base_url: str | None = None) -> dict:
    """ASYNC httpx call. Critical: the retrieval tools run INSIDE the
    same FastAPI process that serves /api/search + friends. A sync
    client call from an async request handler deadlocks the event
    loop (tool waits on the socket, uvicorn can't accept the new
    inbound request because its event loop is blocked). Async
    yields while waiting, so the inbound handler runs and replies.

    4xx / 5xx responses are NOT raised — they're returned as
    `{error: "<body>"}` so the LLM sees the server's message as a
    tool result and can retry with a corrected query. Raising would
    crash the whole turn on one bad SQL syntax mistake.
    """
    url = (base_url or _default_base_url()) + path
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        resp = await client.get(url, params=params or {})
        if resp.status_code >= 400:
            return _error_payload(resp)
        return resp.json()


async def _post(path: str, *, json: dict, base_url: str | None = None) -> dict:
    url = (base_url or _default_base_url()) + path
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        resp = await client.post(url, json=json)
        if resp.status_code >= 400:
            return _error_payload(resp)
        return resp.json()


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


async def search_emails(query: str, date_from: str = "", date_to: str = "", top_k: int = 10) -> dict:
    """Search for messages by relevance (semantic + BM25 blend).

    Use this for "what did we decide about X" / "find messages
    mentioning Y" questions. Empty `date_from` / `date_to` skip the
    date filter. `top_k` caps the number of threads returned; default
    10 is a sensible middle ground.

    Returns {"results": [{thread_id, cite_ref, subject, participants,
    score, matches: [...]}]}. Each match carries the top-hit message
    id + a short snippet; call `get_thread` for full bodies.
    """
    params: dict[str, Any] = {"q": query, "k": top_k}
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    data = await _get("/api/search", params=params)
    # cite_ref = full thread_id. Earlier we used the first 8 hex chars
    # as a shorter shorthand, but that's a 32-bit namespace and
    # collisions occur in real mailboxes (e.g. two emails arriving in
    # the same second from the same sender). The UI shortens chips
    # visually anyway, so there's no benefit to truncating here.
    for row in data.get("results", []):
        row.setdefault("cite_ref", row.get("thread_id") or "")
    return data


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
    data = await _get("/api/query", params=params)
    for row in data.get("results", []):
        row.setdefault("cite_ref", row.get("thread_id") or "")
    return data


# ── get_thread ─────────────────────────────────────────────────────


async def get_thread(thread_id: str) -> dict:
    """Fetch every message in a thread with body text + attachment
    manifest. Bodies clipped to 20k chars (an `original_chars` field
    tells you how much was dropped). Use AFTER search/query when you
    need the actual words a message contained."""
    data = await _get(f"/api/thread/{thread_id}")
    for msg in data.get("messages", []):
        original = msg.get("body_text") or ""
        clipped = _clip(original, THREAD_BODY_CHAR_CAP)
        if clipped != original:
            msg["body_text"] = clipped
            msg["original_chars"] = len(original)
            msg["body_text_truncated"] = True
    return data


# ── sql_query ──────────────────────────────────────────────────────


async def sql_query(query: str) -> dict:
    """Run a read-only SELECT. The server's same safety gate used by
    chat mode applies: only SELECT/WITH, no DDL / DML, no introspection
    of pg_catalog / information_schema, 500-row + 10s timeout.

    Use for aggregations, multi-field OR, JOINs, NOT EXISTS, relative-
    date arithmetic — anything search_emails / query_emails can't
    express. Cells longer than 8000 chars are clipped."""
    data = await _post("/api/sql", json={"query": query})
    # Clip huge cells so one row with a mega body_text doesn't blow
    # the model's context. Match the TS-side cap for consistency.
    clipped_rows: list[list] = []
    for row in data.get("rows", []):
        clipped = []
        for cell in row:
            if isinstance(cell, str) and len(cell) > SQL_CELL_CHAR_CAP:
                clipped.append(_clip(cell, SQL_CELL_CHAR_CAP))
            else:
                clipped.append(cell)
        clipped_rows.append(clipped)
    data["rows"] = clipped_rows
    return data


# ── get_attachment ─────────────────────────────────────────────────


_ATTACHMENT_VALID_MODES = ("meta", "text", "rendered_pages")


async def get_attachment(attachment_id: int, mode: str = "text") -> dict:
    """Fetch an email attachment in one of three shapes:

    - `meta`: filename + mime_type + size_bytes + thread_id, no body.
      Cheap, useful for "what's attached?" queries before deciding
      whether to read.
    - `text`: extracted text (PDF / docx / OCR'd images). Run this
      when you want the attachment's words.
    - `rendered_pages`: PDF pages rasterized to base64-PNG. Heavy
      (multiple MB per page), token-expensive — only use when text
      extraction is empty or unhelpful (scans, complex layouts) AND
      the model can read images.

    `attachment_id` comes from `get_thread`'s `attachments[*].id`."""
    if mode not in _ATTACHMENT_VALID_MODES:
        return {"error": f"invalid mode {mode!r}; must be one of {list(_ATTACHMENT_VALID_MODES)}"}
    if mode == "meta":
        return await _get(f"/api/attachment/{attachment_id}/meta")
    if mode == "text":
        return await _get(f"/api/attachment/{attachment_id}/text")
    return await _get(f"/api/attachment/{attachment_id}/render_pages")


# ── ADK tool factory ───────────────────────────────────────────────


def build_retrieval_tools() -> list:
    """Wrap each retrieval function as an ADK FunctionTool and return
    the list the Retriever / root agent consumes via `tools=[...]`.
    Imported lazily so non-ADK test contexts don't pay the ADK
    import cost just to call the functions directly."""
    from google.adk.tools import FunctionTool

    return [
        FunctionTool(search_emails),
        FunctionTool(query_emails),
        FunctionTool(get_thread),
        FunctionTool(sql_query),
        FunctionTool(get_attachment),
    ]
