"""Cheap reachability probes for the four services the integration
tests require. Used by `conftest.live_stack` to skip the whole
suite cleanly when the operator hasn't brought everything up.

Each probe is a single short HTTP/TCP attempt — never blocks the
test session for more than a few seconds even if all four are dead.
"""

from __future__ import annotations

import socket

import httpx

# Keep these in one place so a relocation (e.g. moving the MCP port)
# doesn't have to grep test files.
GMAIL_FASTAPI_URL = "http://127.0.0.1:8090"
MCP_TOOLS_URL = "http://127.0.0.1:7878"
CLAUDEBOX_URL = "http://127.0.0.1:8765"
POSTGRES_HOST = "127.0.0.1"
POSTGRES_PORT = 5544


def _http_endpoint_up(url: str, timeout: float) -> bool:
    """True if `url` responds at all (any status). We're testing
    reachability, not correctness — a 404 still proves the daemon
    is listening."""
    try:
        with httpx.Client(timeout=timeout) as client:
            client.get(url)
        return True
    except (httpx.HTTPError, OSError):
        return False


def _tcp_port_up(host: str, port: int, timeout: float) -> bool:
    """Plain TCP probe for non-HTTP services like Postgres."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_endpoint_up(url: str, timeout: float = 2.0) -> bool:
    """Public reachability probe. Currently HTTP-only — Postgres uses
    `_tcp_port_up` directly via `probe_all`."""
    return _http_endpoint_up(url, timeout)


def probe_all(timeout: float = 2.0) -> dict[str, bool]:
    """Return a name -> up? dict for every dependency. The conftest
    fixture turns this into a skip message when anything is down."""
    return {
        "gmail-search-fastapi": _http_endpoint_up(f"{GMAIL_FASTAPI_URL}/", timeout),
        "mcp-tools": _http_endpoint_up(f"{MCP_TOOLS_URL}/mcp", timeout),
        "claudebox": _http_endpoint_up(f"{CLAUDEBOX_URL}/health", timeout),
        "postgres": _tcp_port_up(POSTGRES_HOST, POSTGRES_PORT, timeout),
    }


def first_missing(probe_result: dict[str, bool]) -> list[str]:
    """Names of services that came back False, for skip-message use."""
    return [name for name, up in probe_result.items() if not up]
