"""Tests for the end-of-turn auto-publish sweep.

Exercises `auto_publish_unpublished_files` against a temp filesystem
with a fake DB connection. We never import psycopg here — the
`save_artifact` helper is monkeypatched so we can capture what
would-have-been-inserted, and the dedup query is served by an
in-memory `_FakeConn`.
"""

from __future__ import annotations

import os
import time

from gmail_search.agents import auto_publish

# ── Fakes ──────────────────────────────────────────────────────────


class _FakeRow(dict):
    """psycopg-style row supporting both dict-key and attribute
    access. The dedup query reads `r["name"]`, so dict access is
    sufficient — but matching the project's existing fakes keeps the
    pattern uniform if the helper changes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    """In-memory stand-in for psycopg. Returns the configured
    `existing_names` from any SELECT (only the dedup query reads),
    and is a no-op for everything else. `save_artifact` is monkeypatched
    away in tests, so this conn is never the path artifacts actually
    flow through."""

    def __init__(self, existing_names: list[str] | None = None):
        self.existing_names = existing_names or []
        self.queries: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple = ()):
        self.queries.append((sql, params))
        rows = [_FakeRow(name=n) for n in self.existing_names]
        return _FakeCursor(rows)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


def _install_fake_save_artifact(monkeypatch):
    """Capture every `save_artifact` call so tests can assert the
    sweep's insertion behaviour without touching a real DB. Returns
    the calls list; the next id assigned increments from 100."""
    calls: list[dict] = []
    counter = {"next_id": 100}

    def fake_save_artifact(conn, *, session_id, name, mime_type, data, meta=None):
        calls.append(
            {
                "session_id": session_id,
                "name": name,
                "mime_type": mime_type,
                "size": len(data),
                "meta": dict(meta or {}),
            }
        )
        counter["next_id"] += 1
        return counter["next_id"]

    monkeypatch.setattr(auto_publish, "save_artifact", fake_save_artifact)
    return calls


def _set_publish_roots(monkeypatch, tmp_path, *, workspace_name="ws-A", conv_id="conv-A"):
    """Repoint the module's filesystem roots at `tmp_path` so tests
    can build a fake workspace + scratch tree without touching the
    repo's real `deploy/claudebox/workspaces/`."""
    workspace_root = tmp_path / "workspaces"
    scratch_root = tmp_path / "scratch"
    monkeypatch.setattr(auto_publish, "_PUBLISH_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setattr(auto_publish, "_PUBLISH_SCRATCH_ROOT", str(scratch_root))
    workspace_dir = workspace_root / workspace_name
    scratch_dir = scratch_root / conv_id
    workspace_dir.mkdir(parents=True)
    scratch_dir.mkdir(parents=True)
    return workspace_dir, scratch_dir, workspace_name, conv_id


def _write_file(path, *, contents: bytes = b"x" * 200, mtime: float | None = None) -> None:
    path.write_bytes(contents)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


# ── Tests ──────────────────────────────────────────────────────────


def test_publishes_file_written_during_turn(monkeypatch, tmp_path):
    """A fresh file in the workspace must end up in the published
    list with the right name, mime, and size."""
    ws_dir, _scratch, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    calls = _install_fake_save_artifact(monkeypatch)
    turn_started = time.time() - 1
    _write_file(ws_dir / "report.csv", contents=b"a,b\n1,2\n" * 30)

    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace=ws,
        conversation_id=conv,
        turn_started_at=turn_started,
    )

    assert len(out) == 1
    assert out[0]["name"] == "report.csv"
    assert out[0]["mime_type"].startswith("text/csv") or out[0]["mime_type"] == "text/csv"
    assert out[0]["size"] == 240
    assert calls and calls[0]["meta"]["auto_published"] is True
    assert calls[0]["meta"]["source_path"] == "report.csv"


def test_skips_file_written_before_turn(monkeypatch, tmp_path):
    """A file with mtime older than `turn_started_at` is pre-existing
    scratch and must be left alone."""
    ws_dir, _scratch, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    calls = _install_fake_save_artifact(monkeypatch)
    turn_started = time.time()
    _write_file(ws_dir / "old.csv", contents=b"a" * 200, mtime=turn_started - 600)

    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace=ws,
        conversation_id=conv,
        turn_started_at=turn_started,
    )

    assert out == []
    assert calls == []


def test_skips_files_already_published_by_name(monkeypatch, tmp_path):
    """If `agent_artifacts` already has a row whose `name` matches
    the candidate's basename, the sweep treats it as already published
    and skips the insert."""
    ws_dir, _scratch, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    calls = _install_fake_save_artifact(monkeypatch)
    turn_started = time.time() - 1
    _write_file(ws_dir / "plot.png", contents=b"\x89PNG" + b"x" * 200)

    conn = _FakeConn(existing_names=["plot.png"])
    out = auto_publish.auto_publish_unpublished_files(
        conn,
        session_id="sess-1",
        workspace=ws,
        conversation_id=conv,
        turn_started_at=turn_started,
    )

    assert out == []
    assert calls == []


def test_skips_empty_or_tiny_files(monkeypatch, tmp_path):
    """Files smaller than `min_bytes_per_file` are likely empty
    placeholders; never publish them."""
    ws_dir, _scratch, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    calls = _install_fake_save_artifact(monkeypatch)
    turn_started = time.time() - 1
    _write_file(ws_dir / "empty.txt", contents=b"")
    _write_file(ws_dir / "tiny.txt", contents=b"hi")

    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace=ws,
        conversation_id=conv,
        turn_started_at=turn_started,
    )

    assert out == []
    assert calls == []


def test_skips_oversized_file_with_warning(monkeypatch, tmp_path, caplog):
    """Files larger than `max_bytes_per_file` exceed the BYTEA cap
    and must be skipped with a warning, but the sweep must continue
    processing other files."""
    import logging

    ws_dir, _scratch, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    calls = _install_fake_save_artifact(monkeypatch)
    turn_started = time.time() - 1
    big = ws_dir / "big.bin"
    _write_file(big, contents=b"x" * 4096)
    small = ws_dir / "small.txt"
    _write_file(small, contents=b"x" * 200)

    with caplog.at_level(logging.WARNING, logger=auto_publish.logger.name):
        out = auto_publish.auto_publish_unpublished_files(
            _FakeConn(),
            session_id="sess-1",
            workspace=ws,
            conversation_id=conv,
            turn_started_at=turn_started,
            max_bytes_per_file=1024,
        )

    # Only the small file made it through.
    assert [r["name"] for r in out] == ["small.txt"]
    assert len(calls) == 1
    assert any("big.bin" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_skips_dotfiles_and_runner_scaffolding(monkeypatch, tmp_path):
    """The hard-coded skip list: dotfiles, dotted dirs, run.py,
    inputs.json, _manifest.jsonl, and the artifacts/_manifest.jsonl
    sandbox manifest. Everything else passes."""
    ws_dir, _scratch, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    calls = _install_fake_save_artifact(monkeypatch)
    turn_started = time.time() - 1
    # Skipped files:
    _write_file(ws_dir / ".env", contents=b"x" * 200)
    _write_file(ws_dir / "run.py", contents=b"print(1)\n" * 30)
    _write_file(ws_dir / "inputs.json", contents=b"{" + b"x" * 200 + b"}")
    _write_file(ws_dir / "_manifest.jsonl", contents=b"x" * 200)
    artifacts_dir = ws_dir / "artifacts"
    artifacts_dir.mkdir()
    _write_file(artifacts_dir / "_manifest.jsonl", contents=b"x" * 200)
    hidden_dir = ws_dir / ".mpl"
    hidden_dir.mkdir()
    _write_file(hidden_dir / "cache.bin", contents=b"x" * 200)
    pycache = ws_dir / "__pycache__"
    pycache.mkdir()
    _write_file(pycache / "x.pyc", contents=b"x" * 200)
    # Kept file:
    _write_file(ws_dir / "real_output.csv", contents=b"a,b\n1,2\n" * 30)

    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace=ws,
        conversation_id=conv,
        turn_started_at=turn_started,
    )

    assert [r["name"] for r in out] == ["real_output.csv"]
    assert len(calls) == 1


def test_scans_both_workspace_and_scratch(monkeypatch, tmp_path):
    """Both publishable roots must be walked. Files in either show
    up in the published list."""
    ws_dir, scratch_dir, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    _install_fake_save_artifact(monkeypatch)
    turn_started = time.time() - 1
    _write_file(ws_dir / "from_workspace.csv", contents=b"a" * 200)
    _write_file(scratch_dir / "from_scratch.png", contents=b"\x89PNG" + b"y" * 200)

    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace=ws,
        conversation_id=conv,
        turn_started_at=turn_started,
    )

    names = sorted(r["name"] for r in out)
    assert names == ["from_scratch.png", "from_workspace.csv"]


def test_caps_at_max_files(monkeypatch, tmp_path):
    """`max_files` is a hard ceiling that prevents a runaway sweep
    from inserting hundreds of artifacts in one turn."""
    ws_dir, _scratch, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    _install_fake_save_artifact(monkeypatch)
    turn_started = time.time() - 1
    for i in range(20):
        _write_file(ws_dir / f"out_{i:02d}.bin", contents=b"x" * 200)

    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace=ws,
        conversation_id=conv,
        turn_started_at=turn_started,
        max_files=3,
    )

    assert len(out) == 3


def test_no_roots_returns_empty(monkeypatch, tmp_path):
    """When neither workspace nor conversation_id is set, the sweep
    is a no-op. (And when both are set but the dirs don't exist,
    same — the helper returns [] without touching the DB.)"""
    monkeypatch.setattr(auto_publish, "_PUBLISH_WORKSPACE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setattr(auto_publish, "_PUBLISH_SCRATCH_ROOT", str(tmp_path / "scratch"))
    calls = _install_fake_save_artifact(monkeypatch)

    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace=None,
        conversation_id=None,
        turn_started_at=time.time() - 1,
    )
    assert out == []

    # Both set but dirs missing — also a no-op (no error).
    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace="ghost",
        conversation_id="phantom",
        turn_started_at=time.time() - 1,
    )
    assert out == []
    assert calls == []


def test_save_artifact_failure_does_not_break_sweep(monkeypatch, tmp_path):
    """A transient `save_artifact` failure on one file must not abort
    the sweep — log + skip, then keep going."""
    ws_dir, _scratch, ws, conv = _set_publish_roots(monkeypatch, tmp_path)
    turn_started = time.time() - 1
    _write_file(ws_dir / "bad.bin", contents=b"x" * 200)
    _write_file(ws_dir / "good.bin", contents=b"y" * 200)

    counter = {"next_id": 500}

    def flaky_save(conn, *, session_id, name, mime_type, data, meta=None):
        if name == "bad.bin":
            raise RuntimeError("simulated transient db failure")
        counter["next_id"] += 1
        return counter["next_id"]

    monkeypatch.setattr(auto_publish, "save_artifact", flaky_save)

    out = auto_publish.auto_publish_unpublished_files(
        _FakeConn(),
        session_id="sess-1",
        workspace=ws,
        conversation_id=conv,
        turn_started_at=turn_started,
    )

    names = [r["name"] for r in out]
    assert names == ["good.bin"]


def test_build_auto_publish_footer_format():
    """The footer string is the user-facing chip list. Empty in →
    empty out; populated in → divider + intro line + one bullet per
    artifact."""
    assert auto_publish.build_auto_publish_footer([]) == ""
    rows = [
        {"id": 42, "name": "report.xlsx", "mime_type": "application/vnd.ms-excel", "size": 1024},
        {"id": 43, "name": "plot.png", "mime_type": "image/png", "size": 2048},
    ]
    out = auto_publish.build_auto_publish_footer(rows)
    assert "---" in out
    assert "[art:42] **report.xlsx**" in out
    assert "[art:43] **plot.png**" in out
    # Intro line must explain why the chips are appearing.
    assert "produced during this analysis" in out
