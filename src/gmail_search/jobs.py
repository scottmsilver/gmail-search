"""Thin helper for spawning detached gmail-search subprocesses from the
HTTP layer. Isolated here so the server doesn't grow its own process
management concerns and so the call-site stays small + testable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Background workers (backfill embed, reindex/ScaNN builds, crawl, etc.) are
# spawned at a lower CPU priority so the web server (`gmail-search serve`, which
# is NOT spawned through here and stays at nice 0) can always preempt them. On
# an idle box the workers still use every core; under contention serve wins.
# Absolute via setpriority (not os.nice, which increments) so the supervisor →
# worker spawn chain doesn't compound the niceness. Override with GMS_DAEMON_NICE.
_DAEMON_NICE = int(os.environ.get("GMS_DAEMON_NICE", "10"))

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


def argv_matches_job(argv: list[str], job_id: str) -> bool:
    """Decide whether `argv` is an already-running gmail-search daemon for
    `job_id`. Pure function so it's trivially testable — the `/proc`
    walker delegates to this.

    `job_id` is either a bare CLI subcommand (e.g. `"watch"`) or a
    per-user composite key like `"watch:<user_id>"`. The composite case
    matches argv whose subcommand is `watch` AND whose `--email <addr>`
    flag is present (the supervisor then disambiguates by user_id at
    the caller — see `is_daemon_running` for that path).

    Matches only the CLI subcommand itself, not arbitrary substrings — so
    `supervise` won't mistake a shell running `pgrep -f gmail-search
    summarize` for the summarize daemon.
    """
    if ":" in job_id:
        subcommand, _user_suffix = job_id.split(":", 1)
    else:
        subcommand = job_id
    # gmail-search <subcommand> [args] — subcommand is the first positional
    # after any interpreter + script path. argv[0] is either the
    # gmail-search shim or `python`; `python -m gmail_search.cli <cmd>`
    # puts the cmd 3 slots in. Scan for the cmd token directly.
    for i, tok in enumerate(argv):
        # Skip interpreter + script/module tokens.
        if tok in ("python", "python3") or tok.endswith("/python") or tok.endswith("/python3"):
            continue
        if tok.endswith("gmail_search.cli") or tok.endswith("gmail-search"):
            continue
        if tok == "-m":
            continue
        # First non-skip token — that's the subcommand.
        return tok == subcommand
    return False


def is_daemon_running(job_id: str, *, email: str | None = None) -> bool:
    """Scan `/proc` for any process running `gmail-search <job_id>`.
    Zombies (status 'Z') and our own process are ignored.

    For per-user job_ids (`"watch:<uid>"`), pass the user's `email` so
    the match also requires `--email <addr>` in argv — otherwise a
    `gmail-search watch` for one user would falsely flag every other
    user's watch as already-running.

    Returns True if at least one matching process exists. The
    supervisor uses this as a pre-spawn guard — heartbeat staleness is
    necessary but not sufficient, because a daemon blocked in a long
    HTTP call can still be a live process we don't want to duplicate.
    """
    import os

    self_pid = os.getpid()
    proc_root = Path("/proc")
    try:
        dirs = proc_root.iterdir()
    except OSError:
        return False
    for d in dirs:
        name = d.name
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == self_pid:
            continue
        try:
            cmdline_bytes = (d / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        if not cmdline_bytes:
            continue
        argv = cmdline_bytes.rstrip(b"\x00").split(b"\x00")
        argv_str = [a.decode("utf-8", errors="replace") for a in argv]
        if not argv_matches_job(argv_str, job_id):
            continue
        # Per-user disambiguation: same subcommand for two users is
        # one process per user, identified by `--email <addr>`. Without
        # this check, scott's running `gmail-search watch` would mark
        # silvershabbat's `watch:<uid>` as already-running and the
        # supervisor would never spawn it.
        if email is not None:
            try:
                idx = argv_str.index("--email")
                if idx + 1 >= len(argv_str) or argv_str[idx + 1] != email:
                    continue
            except ValueError:
                # No --email at all in argv → not the per-user variant.
                continue
        # Skip zombies — they're reaped imminently.
        try:
            status = (d / "status").read_text()
        except (OSError, PermissionError):
            continue
        # `State: Z (zombie)` — the first word after "State:" tells us.
        for line in status.splitlines():
            if line.startswith("State:"):
                state_char = line.split()[1] if len(line.split()) > 1 else ""
                if state_char == "Z":
                    break  # zombie, skip
                return True
        # No State line? be conservative and count as alive.
    return False


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
        # Lower CPU priority (absolute) so serve preempts under contention.
        preexec_fn=lambda: os.setpriority(os.PRIO_PROCESS, 0, _DAEMON_NICE),
    )
    logger.info(f"spawned pid={proc.pid} argv={argv} log={log_path}")
    return proc.pid
