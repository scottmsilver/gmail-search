"""Thin helper for spawning detached gmail-search subprocesses from the
HTTP layer. Isolated here so the server doesn't grow its own process
management concerns and so the call-site stays small + testable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def gmail_search_command() -> list[str]:
    """Return argv[0..] that invokes this repo's gmail-search CLI. Uses
    the installed console script when present (picks up editable-install
    source), falls back to `python -m gmail_search.cli`.
    """
    installed = shutil.which("gmail-search")
    if installed:
        return [installed]
    return [sys.executable, "-m", "gmail_search.cli"]


def spawn_detached(argv: Iterable[str], log_path: Path) -> int:
    """Spawn a detached subprocess whose stdout/stderr land in log_path.
    Returns the child's PID. The parent returns immediately; the child
    outlives the request.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    proc = subprocess.Popen(
        list(argv),
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    logger.info(f"spawned pid={proc.pid} argv={argv} log={log_path}")
    return proc.pid
