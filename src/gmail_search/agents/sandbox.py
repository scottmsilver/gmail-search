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
import shutil  # noqa: F401  # used inside _cleanup_workdir; formatter strips otherwise
import subprocess
import tempfile
import time
import uuid  # noqa: F401  # used in execute_in_sandbox; formatter has stripped this before
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
    """One-shot container invocation. `evidence` is JSON-serialised
    into the workdir and exposed to the snippet as `evidence` (pandas
    DataFrame via the preamble — see `_serialize_evidence_for_sandbox`
    for accepted shapes). `db_dsn` — if set — is passed as an env var
    the preamble uses to open a read-only PG connection."""

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


def _serialize_evidence_for_sandbox(evidence: Any) -> Any:
    """Coerce `evidence` into a JSON-safe shape the preamble understands.

    We *deliberately* do not pickle here: evidence rows contain Gmail
    bodies / sender strings / untrusted user data. `pickle.load` on
    that is a remote code execution gadget the moment a malicious
    email slips in. JSON is safe.

    Accepted inputs:
      - None → None (preamble → empty DataFrame)
      - pandas DataFrame → list of record dicts
      - dict (assumed column-oriented: {col: [values...]}) → passthrough
      - list (assumed record-oriented: [{col: v, ...}, ...]) → passthrough
      - anything else → str(...) fallback, wrapped as an opaque string
        so the snippet at least sees SOMETHING rather than silently
        getting an empty frame.
    """
    if evidence is None:
        return None
    # Lazy import so `sandbox.py` stays importable in environments
    # without pandas (e.g. the cost/skills test harness).
    try:
        import pandas as pd  # type: ignore

        if isinstance(evidence, pd.DataFrame):
            return evidence.to_dict(orient="records")
    except Exception:
        pass
    if isinstance(evidence, (dict, list)):
        return evidence
    return str(evidence)


# `runpy` creates a fresh module namespace per call — the preamble's


# `runpy` creates a fresh module namespace per call — the preamble's
# names (evidence, db, save_artifact, pd, plt, etc.) wouldn't be
# visible to the snippet. Instead we exec both in the SAME dict so the
# snippet runs with the preamble's globals already populated.
#
# `__name__` is forced to `"__main__"` for the SHARED dict so that any
# `if __name__ == "__main__":` block the analyst writes in `run.py`
# fires (most plotting / CLI-style snippets are written that way). It
# also means the preamble must intentionally NOT contain a main guard
# of its own — anything the preamble wraps in `if __name__ ==
# "__main__":` would actually run here (because we set __name__ to
# "__main__") which is what we want, BUT the next person editing the
# preamble might wrap setup code in such a guard "for safety" and
# create a footgun: in normal `import` use the preamble setup
# wouldn't fire, but here it would, producing inconsistent behaviour
# for whoever copy-pastes the preamble into another harness. Keep
# the preamble as flat top-level statements; if you need conditional
# setup, branch on something other than `__name__`.
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


def _count_remaining_files(workdir: Path) -> int:
    """Count files left under `workdir` after a cleanup attempt. Used
    only to size the warning message, so failures here are themselves
    swallowed (a broken count must not turn a cleanup-warning into a
    cleanup-exception — caller's contract is "never raise")."""
    n = 0
    try:
        for _root, _dirs, files in os.walk(workdir):
            n += len(files)
    except Exception:
        pass
    return n


def _cleanup_workdir(workdir: Path) -> None:
    """Best-effort `shutil.rmtree(workdir)`. If something fails (e.g.
    the container left files owned by uid=1000 inside its user
    namespace that the host can't unlink) we WARN with the leftover
    file count instead of swallowing silently — silent swallow lets
    `/tmp` accumulate analyst workdirs across runs.

    Caller does not depend on cleanup success; we never raise.
    """
    try:
        shutil.rmtree(workdir)
        return
    except OSError as e:
        # rmtree may have removed some of the tree before failing —
        # count what's left so the warning conveys "is this getting
        # worse over time?".
        leftover = _count_remaining_files(workdir)
        logger.warning(
            "sandbox workdir cleanup left %d file(s) at %s: %s",
            leftover,
            workdir,
            e,
        )


def _kill_container(name: str) -> None:
    """Best-effort `docker kill <name>`. Used when our outer
    `subprocess.run(timeout=…)` fires — SIGKILL on the docker CLI does
    NOT kill the container the daemon is already running, so without
    this the container orphans, pins a CPU forever, and `--rm` never
    triggers. Failure here is logged and swallowed: the caller is
    already on the timeout path."""
    try:
        subprocess.run(
            ["docker", "kill", name],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("docker kill %s failed: %s", name, e)


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
    # Evidence goes in as JSON — see `_serialize_evidence_for_sandbox`
    # for the safety rationale (pickle.load on untrusted input = RCE).
    with (workdir / "inputs.json").open("w", encoding="utf-8") as fh:
        json.dump({"evidence": _serialize_evidence_for_sandbox(req.evidence)}, fh)
    # Container runs as uid=1000 in its own user namespace. The host
    # tmpdir is owned by the caller, so we just need owner+group rwx;
    # 0o770 gives the container's mapped uid write access without
    # exposing the workdir to "other" on the host. (Was 0o777 — there
    # is no shared user that needs world-write.)
    os.chmod(workdir, 0o770)
    os.chmod(workdir / "artifacts", 0o770)

    # Name the container so we can `docker kill` it from the timeout
    # branch — see _kill_container().
    container_name = f"gmail-analyst-{uuid.uuid4().hex[:12]}"

    # Build the docker run argv. Kept explicit so a reader can see
    # exactly which guardrails are applied; every flag matters.
    argv: list[str] = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
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
        # subprocess.run only killed the docker CLI; the container is
        # still running in the daemon and would leak forever. Send it
        # a kill so --rm fires.
        _kill_container(container_name)
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
    _cleanup_workdir(workdir)

    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
        wall_ms=wall_ms,
        timed_out=timed_out,
        oom_killed=oom_killed,
    )
