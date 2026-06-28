"""Lightweight request/trace correlation — one id threaded across hops
(MCP tool call -> httpx -> serve request -> Postgres) so a single grep joins
them. Zero dependencies (stdlib contextvars/uuid/logging).

The id is W3C trace-id shaped (32 lowercase hex) and we propagate it as a
`traceparent` header, so if this ever outgrows a single host you can adopt
OpenTelemetry and it will pick up the same ids for free. No OTel dependency
today — full distributed tracing is overkill for one host (see the
observability-reuse research)."""

from __future__ import annotations

import contextvars
import logging
import uuid

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("gms_trace_id", default=None)

# Standard header name we both emit and accept.
TRACEPARENT_HEADER = "traceparent"


def new_trace_id() -> str:
    """A fresh 32-hex trace id (W3C trace-id shape)."""
    return uuid.uuid4().hex


def current_trace_id() -> str | None:
    return _trace_id.get()


def set_trace_id(tid: str | None) -> None:
    _trace_id.set(tid or None)


def ensure_trace_id() -> str:
    """Return the current trace id, minting + setting one if absent."""
    tid = _trace_id.get()
    if not tid:
        tid = new_trace_id()
        _trace_id.set(tid)
    return tid


def make_traceparent(tid: str | None = None) -> str:
    """Build a W3C `traceparent` value: version-traceid-spanid-flags."""
    tid = tid or ensure_trace_id()
    span = uuid.uuid4().hex[:16]
    return f"00-{tid}-{span}-01"


def trace_id_from_header(header: str | None) -> str | None:
    """Extract the 32-hex trace id from a W3C `traceparent`, or accept a bare
    hex id (X-Request-Id style). Returns None if unparseable."""
    if not header:
        return None
    h = header.strip()
    parts = h.split("-")
    if len(parts) >= 3 and len(parts[1]) == 32 and _is_hex(parts[1]):
        return parts[1].lower()
    cand = h.replace("-", "")
    if 8 <= len(cand) <= 64 and _is_hex(cand):
        return cand.lower()
    return None


def _is_hex(s: str) -> bool:
    return bool(s) and all(c in "0123456789abcdefABCDEF" for c in s)


class TraceIdFilter(logging.Filter):
    """Stamp every LogRecord with `trace_id` (or '-') so any formatter can
    include it. Referenced by gmail_search.log_config's dictConfig."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = current_trace_id() or "-"
        return True
