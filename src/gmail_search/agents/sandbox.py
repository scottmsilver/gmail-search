"""Local Docker sandbox for Analyst-authored Python snippets.

Flow:
  1. Caller assembles `SandboxRequest(code, evidence=None, db_dsn=None,
     timeout_seconds=60)` and hands it to `execute_in_sandbox()`.
  2. We materialise a fresh tmp workdir, drop the snippet as `run.py`,
     pickle `evidence` to `inputs.pkl`, and launch a container from
     the `gmail-search-analyst` image built from `sandbox/Dockerfile`.
  3. The container runs `python -c '<preamble>; exec(run.py)'` — the
     preamble loads `evidence`, opens `db`, exposes `save_artifact`.
  4. On exit, we read stdout/stderr, sweep `artifacts/` via
     `_manifest.jsonl`, and return `SandboxResult`. Caller uploads any
     artifacts to the DB and surfaces stdout/stderr in the next
     Analyst LLM turn.

Resource caps applied via `docker run` flags (not the image), so
they're tunable without rebuilds:
  --network=none            no outbound network
  --read-only               root FS frozen; only /work (tmpfs) is writable
  --memory=<cap>            hard RAM ceiling
  --pids-limit=64           belt on fork bombs
  --cpus=1.0                one core, wall-clock limited by --timeout
  --user 1000:1000          non-root, matches the image's analyst uid

The default IMAGE_NAME / tag is kept in one place so CI and the
shipped helper rebuild off the same ref.

**Not in scope for this module**: persisting artifacts to Postgres
(that's the orchestrator's job), matching artifact ids to [art:<id>]
citations, or retrying failed runs. This module just shuttles bytes
in and out of one container run.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# The image built from sandbox/Dockerfile. Match exactly — the
# Dockerfile COPY path (/opt/analyst/preamble.py) is part of this
# contract.
IMAGE_NAME = "gmail-search-analyst:latest"


# Hard ceilings on anything the snippet can consume. Bumping these is a
# deliberate config change, not a per-call knob, because every
# sub-agent turn goes through the same box.
DEFAULT_MEMORY = "512m"
DEFAULT_PIDS_LIMIT = 64
DEFAULT_CPUS = "1.0"
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class SandboxRequest:
    """One-shot container invocation. `evidence` is pickled into the
    workdir and exposed to the snippet as `evidence` (pandas DataFrame
    via the preamble). `db_dsn` — if set — is passed as an env var the
    preamble uses to open a read-only PG connection."""

    code: str
    evidence: Any = None
    db_dsn: str | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    memory: str = DEFAULT_MEMORY
    pids_limit: int = DEFAULT_PIDS_LIMIT
    cpus: str = DEFAULT_CPUS


@dataclass
class SandboxArtifact:
    """One file the snippet produced via `save_artifact(...)`. The
    orchestrator is expected to upload these bytes to the
    `agent_artifacts` table and reply to the Analyst with the
    resulting ids it can cite."""

    name: str
    mime_type: str
    data: bytes


@dataclass
class SandboxResult:
    """Everything the orchestrator needs to show the snippet's output
    to the Analyst agent AND persist artifacts. `exit_code` is 0 on
    clean exit, 124 on timeout (matches GNU coreutils `timeout(1)`),
    137 on OOM-kill, anything else on snippet error."""

    exit_code: int
    stdout: str
    stderr: str
    artifacts: list[SandboxArtifact] = field(default_factory=list)
    wall_ms: int = 0
    timed_out: bool = False
    oom_killed: bool = False


# ── Helpers ──────────────────────────────────────────────────────────


# `runpy` creates a fresh module namespace per call — the preamble's
# names (evidence, db, save_artifact, pd, plt, etc.) wouldn't be
# visible to the snippet. Instead we exec both in the SAME dict so the
# snippet runs with the preamble's globals already populated.
_RUNNER_SCRIPT = """\
ns = {'__name__': '__main__'}
with open('/opt/analyst/preamble.py', 'r') as f:
    exec(compile(f.read(), '/opt/analyst/preamble.py', 'exec'), ns)
with open('/work/run.py', 'r') as f:
    exec(compile(f.read(), '/work/run.py', 'exec'), ns)
"""


def _sweep_artifacts(workdir: Path) -> list[SandboxArtifact]:
    """Walk /work/artifacts/_manifest.jsonl and load each byte-blob into
    a SandboxArtifact. The preamble's `save_artifact(...)` appends one
    line per file to that manifest so we never accidentally pick up
    temp files the snippet wrote for its own scratch use."""
    artifacts_dir = workdir / "artifacts"
    manifest = artifacts_dir / "_manifest.jsonl"
    if not manifest.exists():
        return []
    out: list[SandboxArtifact] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"bad manifest line {line!r}: {e}")
            continue
        name = entry.get("name")
        mime = entry.get("mime_type")
        if not name or not mime:
            continue
        path = artifacts_dir / name
        if not path.is_file():
            logger.warning(f"manifest references missing file {name!r}")
            continue
        out.append(SandboxArtifact(name=name, mime_type=mime, data=path.read_bytes()))
    return out


def image_available() -> bool:
    """True iff the analyst image is already built locally. Used by
    tests to skip cleanly on CI / dev machines that haven't run
    `docker build` yet."""
    try:
        rv = subprocess.run(
            ["docker", "image", "inspect", IMAGE_NAME],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return rv.returncode == 0


# ── Public API ──────────────────────────────────────────────────────


def execute_in_sandbox(req: SandboxRequest) -> SandboxResult:
    """Run one snippet in a fresh ephemeral container. Blocks until the
    container exits or our outer wall-clock (`timeout_seconds + 5`)
    triggers — we add a small grace so Docker's own `--stop-timeout`
    kicks in before we do.

    Never raises on snippet failure; those are reported via
    `exit_code`, `stderr`, `timed_out`, `oom_killed`. It DOES raise on
    infrastructure failure (image missing, docker binary missing).
    """
    if not image_available():
        raise RuntimeError(f"sandbox image {IMAGE_NAME} not built — run `docker build -t {IMAGE_NAME} sandbox/`")

    # Fresh scratch dir per run — avoids any cross-invocation state
    # leakage even if the caller forgets to clean up.
    workdir = Path(tempfile.mkdtemp(prefix="gmail-analyst-"))
    (workdir / "artifacts").mkdir(exist_ok=True)
    (workdir / "run.py").write_text(req.code, encoding="utf-8")
    with (workdir / "inputs.pkl").open("wb") as fh:
        pickle.dump({"evidence": req.evidence}, fh)
    os.chmod(workdir, 0o777)  # container user is uid=1000; make /work writable
    os.chmod(workdir / "artifacts", 0o777)

    # Build the docker run argv. Kept explicit so a reader can see
    # exactly which guardrails are applied; every flag matters.
    argv: list[str] = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        f"--memory={req.memory}",
        f"--pids-limit={req.pids_limit}",
        f"--cpus={req.cpus}",
        "--user",
        "1000:1000",
        "--tmpfs",
        "/tmp:rw,size=64m",
        "-v",
        f"{workdir}:/work:rw",
        "--stop-timeout",
        "2",
        "-e",
        f"ANALYST_DB_DSN={req.db_dsn or ''}",
        IMAGE_NAME,
        "python",
        "-c",
        _RUNNER_SCRIPT,
    ]

    start = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            timeout=req.timeout_seconds + 5,
            text=True,
            errors="replace",
        )
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = (
            (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        )
        stderr = (
            (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        )
        exit_code = 124  # GNU timeout convention

    wall_ms = int((time.time() - start) * 1000)
    # Docker reports 137 for SIGKILL-on-oom. Not perfectly unambiguous
    # (137 also means plain SIGKILL) but the memory cap is the usual
    # cause in this sandbox, so surfacing it as "oom_killed" is
    # accurate more often than not.
    oom_killed = exit_code == 137 and not timed_out

    artifacts = _sweep_artifacts(workdir) if exit_code == 0 else []

    # Best-effort cleanup. A readonly-rootfs leftover container is
    # already removed by --rm; the tmp workdir is ours.
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except OSError:
        pass

    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
        wall_ms=wall_ms,
        timed_out=timed_out,
        oom_killed=oom_killed,
    )
