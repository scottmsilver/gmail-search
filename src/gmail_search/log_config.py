"""One shared logging configuration for every entrypoint (serve, MCP server,
CLI daemons). Human-readable by default; set `GMS_LOG_JSON=1` for JSON lines
(what you want in files / under systemd so they're machine-queryable with jq).

Reuses stdlib `logging.config.dictConfig` + `python-json-logger` (a pure-Python
JSON `logging.Formatter` — zero behavior change to the logging tree, so the
existing `logging.getLogger(__name__)` call sites and printf-style messages keep
working untouched; you just get a JSON envelope + any `extra={...}` fields).

Every line carries `trace_id` (via gmail_search.trace.TraceIdFilter) so logs
across serve/MCP/daemons join on one id."""

from __future__ import annotations

import copy
import logging.config
import os

_TRACE_FILTER = "gmail_search.trace.TraceIdFilter"
# Text format mirrors the daemons' prior format plus a [trace_id] field.
_TEXT_FMT = "%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s"
# JSON: message stays the human string; structured extras (tool, session_id,
# trace_id, …) ride alongside. python-json-logger merges record attrs + extras.
_JSON_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s"


def json_enabled(force: bool | None = None) -> bool:
    if force is not None:
        return force
    return os.environ.get("GMS_LOG_JSON", "") == "1"


def _active_formatter(json_logs: bool) -> dict:
    """Only the formatter actually in use. We deliberately do NOT also define the
    JSON formatter in text mode: dictConfig instantiates every formatter entry,
    so referencing the JSON one would import python-json-logger even in text
    mode and crash if it isn't installed (it's a declared dep, but this keeps a
    code-change-without-install from hard-failing)."""
    if json_logs:
        return {"()": "pythonjsonlogger.json.JsonFormatter", "fmt": _JSON_FMT}
    return {"format": _TEXT_FMT}


def build_dict_config(*, json_logs: bool | None = None, level: str = "INFO") -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"trace": {"()": _TRACE_FILTER}},
        "formatters": {"active": _active_formatter(json_enabled(json_logs))},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "active",
                "filters": ["trace"],
            }
        },
        "root": {"handlers": ["default"], "level": level},
    }


def setup_logging(*, json_logs: bool | None = None, level: str = "INFO") -> None:
    """Apply the shared config. Safe to call from any entrypoint; idempotent
    (disable_existing_loggers=False keeps already-created loggers working)."""
    logging.config.dictConfig(build_dict_config(json_logs=json_logs, level=level))


def uvicorn_log_config(*, json_logs: bool | None = None) -> dict:
    """Uvicorn's own LOGGING_CONFIG with our formatter + trace filter merged in.
    Needed because uvicorn's loggers set `propagate=False`, so the root handler
    never sees uvicorn.access / uvicorn.error — we must reformat them in place."""
    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    json_logs = json_enabled(json_logs)
    cfg.setdefault("filters", {})["trace"] = {"()": _TRACE_FILTER}
    # Our active formatter (text or JSON), carrying trace_id.
    cfg["formatters"]["gms_active"] = _active_formatter(json_logs)
    if json_logs:
        # In JSON mode, reformat uvicorn's own access/error loggers as JSON too.
        for name in ("default", "access"):
            if name in cfg["formatters"]:
                cfg["formatters"][name] = {"()": "pythonjsonlogger.json.JsonFormatter", "fmt": _JSON_FMT}
    # Dedicated handler for OUR app loggers (which propagate to root — e.g. the
    # per-request log line). It uses our formatter, which prints trace_id;
    # uvicorn's own "default" formatter does not. uvicorn's LOGGING_CONFIG has no
    # "root", so without this our app logs would be dropped under uvicorn's config.
    cfg["handlers"]["gms_app"] = {
        "class": "logging.StreamHandler",
        "formatter": "gms_active",
        "filters": ["trace"],
        "stream": "ext://sys.stderr",
    }
    for name, handler in cfg.get("handlers", {}).items():
        if name != "gms_app":
            handler.setdefault("filters", []).append("trace")
    # uvicorn's own loggers set propagate=False, so routing root at gms_app
    # won't double-log them.
    cfg["root"] = {"handlers": ["gms_app"], "level": "INFO"}
    return cfg
