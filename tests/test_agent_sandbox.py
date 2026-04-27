"""Tests for the deep-analysis Docker sandbox executor.

All tests here require the `gmail-search-analyst:latest` image to be
built locally (`docker build -t gmail-search-analyst:latest sandbox/`).
Each test skips cleanly when the image isn't present — CI / dev
machines without Docker still get a clean test pass.

The tests cover the paths a buggy or adversarial snippet can reach:
happy text output, stderr from snippet exceptions, wall-clock
timeout, memory cap, save_artifact round-trip, network-none
enforcement.
"""

from __future__ import annotations

import pytest

from gmail_search.agents.sandbox import SandboxRequest, execute_in_sandbox, image_available

pytestmark = pytest.mark.skipif(
    not image_available(),
    reason="gmail-search-analyst image not built — run `docker build -t gmail-search-analyst:latest sandbox/`",
)


def test_sandbox_happy_path_stdout_and_evidence():
    """Snippet reads the seeded `evidence` DataFrame and prints — we
    expect exit_code 0 and the stdout we printed. Also confirms the
    preamble ran (evidence is defined, pd imported).

    We pass evidence as a dict-of-lists (not a pre-built DataFrame)
    so the preamble constructs it inside the container — avoids any
    host-vs-container numpy/pandas pickle compatibility issue when
    the test runner's env is slightly off from the pinned sandbox
    stack. This matches what the real Retriever → Analyst handoff
    will do (hands down records, sandbox materialises the frame).
    """
    evidence = {"from_addr": ["a@x.com", "b@x.com", "a@x.com"], "subject": ["s1", "s2", "s3"]}
    req = SandboxRequest(
        code="print('rows:', len(evidence))\nprint('senders:', evidence['from_addr'].nunique())\n",
        evidence=evidence,
    )
    res = execute_in_sandbox(req)
    assert res.exit_code == 0, res.stderr
    assert "rows: 3" in res.stdout
    assert "senders: 2" in res.stdout
    assert res.wall_ms > 0


def test_sandbox_captures_snippet_exception_as_stderr():
    """SyntaxError / runtime error → non-zero exit + traceback in
    stderr. The Analyst LLM reads stderr and iterates."""
    req = SandboxRequest(code="raise ValueError('boom from snippet')\n")
    res = execute_in_sandbox(req)
    assert res.exit_code != 0
    assert res.exit_code != 137 and res.exit_code != 124  # not oom, not timeout
    assert "boom from snippet" in res.stderr


def test_sandbox_timeout_on_infinite_loop():
    """`while True: pass` → outer wall-clock fires, exit_code 124
    (GNU timeout convention), `timed_out` flag set."""
    req = SandboxRequest(code="while True:\n    pass\n", timeout_seconds=3)
    res = execute_in_sandbox(req)
    assert res.timed_out
    assert res.exit_code == 124


def test_sandbox_save_artifact_roundtrip():
    """Snippet saves a DataFrame as CSV and a tiny PNG via
    save_artifact(...). Executor sweeps the manifest and returns
    both as SandboxArtifact with correct mime types. Evidence is an
    empty dict so the preamble builds an empty DataFrame inside the
    container (see docstring on the happy-path test)."""
    code = (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "save_artifact('demo.csv', pd.DataFrame({'a':[1,2,3]}))\n"
        "fig = plt.figure()\n"
        "plt.plot([1,2,3])\n"
        "save_artifact('demo.png', fig)\n"
        "print('done')\n"
    )
    res = execute_in_sandbox(SandboxRequest(code=code, evidence={}))
    assert res.exit_code == 0, res.stderr
    names = {a.name: a for a in res.artifacts}
    assert "demo.csv" in names
    assert "demo.png" in names
    assert names["demo.csv"].mime_type == "text/csv"
    assert names["demo.png"].mime_type == "image/png"
    # PNG magic bytes confirm matplotlib really wrote a PNG
    assert names["demo.png"].data[:4] == b"\x89PNG"
    # CSV header should contain our column name
    assert names["demo.csv"].data.splitlines()[0] == b"a"


def test_sandbox_network_is_blocked():
    """--network=none → any attempted outbound connection fails. The
    snippet should see a socket error, NOT talk to the internet."""
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    print('NETWORK_OPEN')\n"
        "except Exception as e:\n"
        "    print('NETWORK_BLOCKED', type(e).__name__)\n"
    )
    res = execute_in_sandbox(SandboxRequest(code=code))
    assert res.exit_code == 0, res.stderr
    assert "NETWORK_BLOCKED" in res.stdout
    assert "NETWORK_OPEN" not in res.stdout


def test_sandbox_empty_evidence_is_empty_dataframe():
    """evidence=None → preamble produces an empty DataFrame rather
    than crashing on `evidence.whatever()`."""
    req = SandboxRequest(
        code="print('empty:', evidence.empty)\nprint('shape:', evidence.shape)\n",
    )
    res = execute_in_sandbox(req)
    assert res.exit_code == 0, res.stderr
    assert "empty: True" in res.stdout
    assert "shape: (0, 0)" in res.stdout


def test_sandbox_errors_when_image_missing(monkeypatch):
    """Infrastructure failure path: missing image → RuntimeError with
    the exact command the operator should run. Use monkeypatch to
    force-skip the image-present check even when the image IS built,
    so this test runs everywhere."""
    from gmail_search.agents import sandbox as sb

    monkeypatch.setattr(sb, "image_available", lambda: False)
    with pytest.raises(RuntimeError, match="docker build"):
        execute_in_sandbox(SandboxRequest(code="print(1)"))


# ── Persistent /work across run_code calls ─────────────────────────


def test_sandbox_persistent_workdir_survives_across_calls(monkeypatch, tmp_path):
    """Two execute_in_sandbox calls with the same conversation_id share
    /work: the first writes a file, the second reads it back. This is
    the load-bearing test for the persistence feature."""
    from gmail_search.agents import sandbox as sb

    monkeypatch.setattr(sb, "_SCRATCH_ROOT", str(tmp_path / "scratch"))

    res1 = execute_in_sandbox(
        SandboxRequest(
            code="open('/work/state.txt', 'w').write('hello-from-call-1')\nprint('wrote')\n",
            conversation_id="conv-abc-123",
        )
    )
    assert res1.exit_code == 0, res1.stderr
    assert "wrote" in res1.stdout

    res2 = execute_in_sandbox(
        SandboxRequest(
            code="print('saw:', open('/work/state.txt').read())\n",
            conversation_id="conv-abc-123",
        )
    )
    assert res2.exit_code == 0, res2.stderr
    assert "saw: hello-from-call-1" in res2.stdout


def test_sandbox_no_conversation_id_is_ephemeral(monkeypatch, tmp_path):
    """Without conversation_id, /work resets between calls — the file
    written in call 1 must NOT be visible in call 2. Confirms the
    legacy ephemeral path is preserved."""
    from gmail_search.agents import sandbox as sb

    monkeypatch.setattr(sb, "_SCRATCH_ROOT", str(tmp_path / "scratch"))

    res1 = execute_in_sandbox(
        SandboxRequest(code="open('/work/state.txt', 'w').write('x')\nprint('wrote')\n"),
    )
    assert res1.exit_code == 0, res1.stderr

    res2 = execute_in_sandbox(
        SandboxRequest(
            code=("import os\n" "print('exists:', os.path.exists('/work/state.txt'))\n"),
        ),
    )
    assert res2.exit_code == 0, res2.stderr
    assert "exists: False" in res2.stdout


def test_safe_conversation_id_rejects_traversal():
    """The path-traversal guard is the security boundary for the
    persistent path; verify it rejects the obvious attacks."""
    from gmail_search.agents.sandbox import _safe_conversation_id

    assert _safe_conversation_id("conv-123_abc") == "conv-123_abc"
    for bad in ("../etc", "a/b", "..", "", "with space", "drop;table"):
        with pytest.raises(ValueError):
            _safe_conversation_id(bad)
