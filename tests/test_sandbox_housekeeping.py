"""Housekeeping tests for the sandbox executor — covers the bits that
are independent of the actual Docker container so they run on every
machine (no `gmail-search-analyst` image required):

* Workdir permissions are tightened to 0o770 (was 0o777).
* `_cleanup_workdir` logs a warning with leftover-file count when
  `shutil.rmtree` fails — we used to swallow with `ignore_errors=True`
  which let dead workdirs accumulate in /tmp silently.
* `_count_remaining_files` walks the tree and returns a count.

The Docker happy-path (which exercises 0o770 end-to-end against a real
container running as uid=1000) lives in `test_agent_sandbox.py` and
auto-skips when the image isn't built.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from gmail_search.agents import sandbox as sb

# ── _cleanup_workdir ──────────────────────────────────────────────────


def test_cleanup_workdir_removes_tree_on_happy_path(tmp_path: Path) -> None:
    """Normal case: rmtree succeeds → no warning, tree gone."""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "run.py").write_text("print(1)")
    (workdir / "artifacts").mkdir()

    sb._cleanup_workdir(workdir)
    assert not workdir.exists()


def test_cleanup_workdir_logs_warning_when_rmtree_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If `shutil.rmtree` raises (the real-world case is container
    leftovers owned by uid=1000 the host can't unlink), we WARN with
    the leftover file count instead of silently swallowing."""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "a.txt").write_text("x")
    (workdir / "b.txt").write_text("y")
    (workdir / "sub").mkdir()
    (workdir / "sub" / "c.txt").write_text("z")

    def _boom(path):  # noqa: ARG001
        raise OSError("simulated: container left files owned by uid 1000")

    monkeypatch.setattr(sb.shutil, "rmtree", _boom)

    with caplog.at_level(logging.WARNING, logger=sb.logger.name):
        sb._cleanup_workdir(workdir)

    # Warning emitted exactly once, mentions the file count and the
    # path so the operator can find the leftovers in /tmp.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected 1 warning, got {len(warnings)}: {[r.getMessage() for r in warnings]}"
    msg = warnings[0].getMessage()
    assert "sandbox workdir cleanup" in msg
    assert "3" in msg, msg  # we wrote three files; the count must show up
    assert str(workdir) in msg


def test_cleanup_workdir_never_raises_on_count_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Even if BOTH rmtree and the count-walk blow up, the helper must
    not propagate — the caller's contract is "best-effort cleanup".
    """
    workdir = tmp_path / "wd"
    workdir.mkdir()

    def _boom(*_a, **_kw):
        raise OSError("rmtree boom")

    def _count_boom(*_a, **_kw):
        raise RuntimeError("count boom")

    monkeypatch.setattr(sb.shutil, "rmtree", _boom)
    monkeypatch.setattr(sb.os, "walk", _count_boom)

    with caplog.at_level(logging.WARNING, logger=sb.logger.name):
        sb._cleanup_workdir(workdir)  # must not raise

    # We still got a warning (with count=0 because the walk bailed).
    assert any("sandbox workdir cleanup" in r.getMessage() for r in caplog.records)


def test_count_remaining_files_walks_tree(tmp_path: Path) -> None:
    """Sanity: `_count_remaining_files` recursively counts files (not
    directories). Leveraged by the warning message above."""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a").write_text("1")
    (root / "b").write_text("2")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c").write_text("3")
    (sub / "d").write_text("4")

    assert sb._count_remaining_files(root) == 4


def test_count_remaining_files_returns_zero_for_missing_dir(tmp_path: Path) -> None:
    """Walking a path that no longer exists returns 0 (not an
    exception). This matters because rmtree may have removed the dir
    entirely before failing on a stragger somewhere else."""
    gone = tmp_path / "does-not-exist"
    assert sb._count_remaining_files(gone) == 0


# ── chmod 0o770 (host-side; the docker test in test_agent_sandbox.py
#    proves the container can still write) ────────────────────────────


def test_workdir_permissions_are_0o770_not_world_writable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`execute_in_sandbox` shouldn't expose its workdir to "other"
    on the host. We monkeypatch `image_available` to True and stub
    `subprocess.run` so the test runs without Docker, capture the
    chmod-applied workdir before cleanup, and assert mode==0o770.
    """
    captured: dict[str, Path] = {}

    monkeypatch.setattr(sb, "image_available", lambda: True)

    real_run = sb.subprocess.run

    def _fake_subprocess_run(argv, **kwargs):  # noqa: ARG001
        # Stash the workdir off the -v flag so we can assert its mode.
        for i, a in enumerate(argv):
            if a == "-v" and i + 1 < len(argv):
                host_path = argv[i + 1].split(":", 1)[0]
                captured["workdir"] = Path(host_path)
                break

        class _R:
            stdout = ""
            stderr = ""
            returncode = 0

        return _R()

    monkeypatch.setattr(sb.subprocess, "run", _fake_subprocess_run)

    # Block the cleanup so the dir survives long enough for our
    # post-execution assertion. (We restore + clean up ourselves at
    # the end.)
    monkeypatch.setattr(sb, "_cleanup_workdir", lambda _p: None)

    try:
        sb.execute_in_sandbox(sb.SandboxRequest(code="print('hi')"))
    finally:
        monkeypatch.setattr(sb.subprocess, "run", real_run)

    workdir = captured.get("workdir")
    assert workdir is not None and workdir.exists(), "stub didn't capture the workdir"

    mode = stat.S_IMODE(os.stat(workdir).st_mode)
    assert mode == 0o770, f"workdir mode is {oct(mode)}, expected 0o770"

    artifacts_mode = stat.S_IMODE(os.stat(workdir / "artifacts").st_mode)
    assert artifacts_mode == 0o770, f"artifacts mode is {oct(artifacts_mode)}, expected 0o770"

    # Real cleanup so we don't leak.
    import shutil as _shutil

    _shutil.rmtree(workdir, ignore_errors=True)
