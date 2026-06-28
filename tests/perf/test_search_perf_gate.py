"""Search performance regression gate.

Commissioned after a debugging session found a single `/api/search` call
taking 14-58s with NOTHING measuring it (the hidden culprit was a
`thread_summary` bulk-IN fetch doing ~9.8s of work inside an otherwise
reasonable `search_threads`). This gate makes that class of bug impossible
to merge silently.

WHAT THIS GATE COVERS
  1. Per-stage search timing  — `test_per_stage_timing` (perf_slow)
     Profiler-style monkeypatch of every stage of `search_threads`; fails
     if any stage regresses past tolerance. This is the stage that would
     have caught the thread_summary bug: the bulk fetch shows up as its own
     line, so a 9.8s sub-stage can't hide inside a 12s total.
  2. End-to-end API latency  — `test_end_to_end_http_latency` (perf_slow)
     Drives the LIVE `/api/search` over HTTP across 4 representative
     queries (clear-winner, tightly-clustered/rerank, BM25/identifier,
     date-filtered) at k=10. Fails if total regresses past tolerance.
  3. Batch & concurrency  — `test_batch_contract_no_raise` (fast gate)
     `search_emails_batch` must return a dict with per-item results and
     NEVER raise, even if one item errors/times out, within a wall budget.
  4. DB query plans  — `test_db_plans_no_seq_scan` (fast gate)
     EXPLAINs the hot bulk-IN queries (thread_summary + bm25 messages) and
     asserts an Index/Bitmap scan, NOT a Seq Scan — directly guarding the
     missing-index / stale-stats regression.

TOLERANCE POLICY
  A metric regresses when it exceeds max(baseline*1.5, baseline+150ms).
  Baseline lives in tests/perf/baseline.json.

FAST vs SLOW
  Fast gate (default `pytest`): batch contract + DB plans. Auto-skip when
  PG / serve env is unavailable, so CI without a DB never hard-fails.
  Slow gate (`pytest -m perf_slow`): per-stage in-process timing (builds
  ONE ~10GB ScaNN engine, ~100s) and live HTTP latency. Opt-in only.

UPDATE THE BASELINE
  pytest tests/perf -m perf_slow --update-baseline
  (regenerates per-stage + end-to-end numbers from a real run, clears the
  needs_blessed_regen flag, and rewrites baseline.json — then commit it.)

See tests/perf/README.md for full docs.
"""

from __future__ import annotations

import time

import harness as H  # tests/perf is on sys.path under pytest's prepend import mode
import pytest

# ── Representative queries ──────────────────────────────────────────────────
# Chosen to exercise distinct code paths (see module docstring item 2).
REPRESENTATIVE_QUERIES: list[dict] = [
    {"name": "clear_winner", "q": "draw request", "kwargs": {"top_k": 10}},
    {"name": "tightly_clustered", "q": "meeting notes update", "kwargs": {"top_k": 10}},
    {"name": "bm25_identifier", "q": "invoice 12345", "kwargs": {"top_k": 10}},
    {
        "name": "date_filtered",
        "q": "draw request",
        "kwargs": {"top_k": 10, "date_from": "2026-01-01", "date_to": "2026-06-30"},
    },
]


# ── baseline-update flag (defined in tests/perf/conftest.py) ────────────────


@pytest.fixture(scope="session")
def update_baseline(request) -> bool:
    return bool(request.config.getoption("--update-baseline"))


@pytest.fixture(scope="session")
def serve_env() -> dict[str, str]:
    return H.resolve_serve_env()


@pytest.fixture(scope="session")
def baseline() -> dict:
    return H.load_baseline()


# ── Shared skip guards ──────────────────────────────────────────────────────


def _require_pg():
    if not H.pg_reachable():
        pytest.skip("Postgres not reachable at 127.0.0.1:5544 — start the paradedb container to enable")


def _require_corpus():
    """Skip unless the test user actually has a populated corpus — perf
    numbers are meaningless on an empty schema."""
    _require_pg()
    with H.closing_connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM thread_summary WHERE user_id = %s",
            (H.test_user_id(),),
        ).fetchone()
    if not row or row["n"] < 100:
        pytest.skip(f"Test user {H.test_user_id()} has no/sparse corpus in this DB — perf gate needs real data")


# ── 4. DB query plans (FAST GATE) ───────────────────────────────────────────


def _sample_thread_ids(conn, n: int) -> list[str]:
    rows = conn.execute(
        "SELECT thread_id FROM thread_summary WHERE user_id = %s LIMIT %s",
        (H.test_user_id(), n),
    ).fetchall()
    return [r["thread_id"] for r in rows]


def _sample_message_ids(conn, n: int) -> list[str]:
    rows = conn.execute(
        "SELECT id FROM messages WHERE user_id = %s LIMIT %s",
        (H.test_user_id(), n),
    ).fetchall()
    return [r["id"] for r in rows]


def _explain_text(conn, sql: str, params: list) -> str:
    rows = conn.execute(f"EXPLAIN (FORMAT TEXT) {sql}", params).fetchall()
    return "\n".join(r[0] for r in rows)


def _assert_indexed_plan(plan: str, table: str):
    """Assert the plan does NOT Seq-Scan `table`. A Seq Scan over a
    multi-hundred-thousand-row table for a ~250-id IN-list is the
    missing-index/stale-stats signature that caused the 9.8s fetch."""
    lowered = plan.lower()
    bad = f"seq scan on {table}".lower()
    assert bad not in lowered, (
        f"Query plan SEQ-SCANS {table} for a bulk-IN fetch — missing index or stale stats. "
        f"This is the exact regression that caused the 9.8s thread_summary fetch.\n\nPlan:\n{plan}"
    )
    # Positive assertion: some index/bitmap access is present for this table.
    has_index = any(
        marker in lowered for marker in ("index scan", "index only scan", "bitmap index scan", "bitmap heap scan")
    )
    assert has_index, f"Plan for {table} bulk-IN uses no index access at all:\n{plan}"


def test_db_plans_no_seq_scan():
    """FAST GATE: hot bulk-IN queries must use Index/Bitmap, not Seq Scan."""
    _require_corpus()
    with H.closing_connection() as conn:
        tids = _sample_thread_ids(conn, 250)
        assert tids, "no thread_ids sampled"
        ph = ",".join(["%s"] * len(tids))
        ts_plan = _explain_text(
            conn,
            f"SELECT * FROM thread_summary WHERE thread_id IN ({ph}) AND user_id = %s",
            tids + [H.test_user_id()],
        )
        _assert_indexed_plan(ts_plan, "thread_summary")

        mids = _sample_message_ids(conn, 200)
        assert mids, "no message_ids sampled"
        mph = ",".join(["%s"] * len(mids))
        msg_plan = _explain_text(
            conn,
            f"SELECT id, thread_id, subject, from_addr, date, body_text "
            f"FROM messages WHERE id IN ({mph}) AND user_id = %s",
            mids + [H.test_user_id()],
        )
        _assert_indexed_plan(msg_plan, "messages")


def _top_plan_cost(plan: str):
    """Parse the estimated total cost (upper bound) of the plan's top node
    from EXPLAIN text, e.g. 'Limit (cost=239033.86..239036.77 rows=25 …)'."""
    import re

    m = re.search(r"cost=[\d.]+\.\.([\d.]+)", plan)
    return float(m.group(1)) if m else None


# Healthy priority-inbox plan is ~239k; the broken correlated-subplan version
# was ~23.8 BILLION. 50M is a wide ceiling that passes the anti-join rewrite
# and screams on any return to the per-row O(messages^2) plan.
PRIORITY_PLAN_COST_CEILING = 50_000_000


def test_priority_inbox_plan_not_quadratic():
    """FAST GATE: the priority-inbox 'everything else' query must not blow up
    into a per-row correlated subplan. That plan hit cost ~23.8 BILLION, ran
    13+ min, and — because serve calls blocking psycopg from its async event
    loop — wedged the WHOLE service. Cap the planner's estimated cost well
    below that. Uses the same NOT EXISTS anti-join the endpoint runs."""
    _require_corpus()
    uid = H.test_user_id()
    pred = (
        "m.labels LIKE %s AND NOT EXISTS ("
        "  SELECT 1 FROM messages m2 WHERE m2.thread_id = m.thread_id AND m2.user_id = m.user_id"
        "  AND ((m2.labels LIKE %s AND m2.labels LIKE %s AND m2.labels LIKE %s) OR (m2.labels LIKE %s)))"
    )
    params = ['%"INBOX"%', '%"IMPORTANT"%', '%"UNREAD"%', '%"INBOX"%', '%"STARRED"%']
    sql = (
        "SELECT ts.thread_id FROM thread_summary ts WHERE ts.user_id = %s AND EXISTS ("
        f"  SELECT 1 FROM messages m WHERE m.thread_id = ts.thread_id AND m.user_id = %s AND ({pred})) "
        "ORDER BY ts.date_last DESC LIMIT 25 OFFSET 0"
    )
    with H.closing_connection() as conn:
        plan = _explain_text(conn, sql, [uid, uid, *params])
    cost = _top_plan_cost(plan)
    assert cost is not None, f"could not parse plan cost:\n{plan}"
    assert cost < PRIORITY_PLAN_COST_CEILING, (
        f"priority-inbox 'everything else' plan cost {cost:,.0f} exceeds "
        f"{PRIORITY_PLAN_COST_CEILING:,} — likely regressed to a per-row correlated subplan "
        f"(the 23.8B / 13-min wedge that froze serve). Plan:\n{plan}"
    )


def test_sql_cost_gate_rejects_pathological():
    """FAST GATE: the agent SQL pre-flight cost gate must REJECT a plan whose
    estimated cost blows past the ceiling (cartesian join / per-row correlated
    subplan) while ALLOWING a normal query. This is the guard that stops an
    agent from running a query dumb enough to wedge the service."""
    _require_corpus()
    from gmail_search import server

    uid = H.test_user_id()
    # Normal query: must be allowed.
    conn = server._open_readonly_connection(user_id=uid)
    try:
        server._reject_if_too_expensive(conn, "SELECT count(*) FROM messages")
    finally:
        conn.close()

    # 3-way cartesian join: astronomically expensive, must be rejected.
    conn = server._open_readonly_connection(user_id=uid)
    try:
        with pytest.raises(server.SqlTooExpensiveError):
            server._reject_if_too_expensive(conn, "SELECT 1 FROM messages a, messages b, messages c")
    finally:
        conn.close()


# ── 3. Batch & concurrency contract (FAST GATE) ─────────────────────────────


@pytest.mark.asyncio
async def test_batch_contract_no_raise(baseline):
    """FAST GATE: search_emails_batch returns a per-item dict and NEVER
    raises, even when an item errors — and finishes within a wall budget.

    Written against the PUBLIC CONTRACT (the bug was one slow item's
    ReadTimeout propagating through asyncio.gather and nuking the whole
    batch). We assert the dict shape + no-raise, not internals like
    DEFAULT_TIMEOUT_SECONDS or the gather call, since the main engineer may
    change those concurrently.

    Does NOT require a live server: we monkeypatch the per-item
    `search_emails` so the test is hermetic and fast. The contract under
    test is the batch wrapper's error isolation + shape, not retrieval.
    """
    import gmail_search.agents.tools as tools

    calls = {"n": 0}

    async def fake_search_emails(*args, **kwargs):
        calls["n"] += 1
        # Item 1 deliberately fails the way a ReadTimeout would surface as a
        # per-item error payload; the batch must still return every result.
        if kwargs.get("query") == "boom":
            return {"error": "simulated upstream timeout"}
        return {"results": [{"thread_id": "t1", "cite_ref": "t1", "matches": []}]}

    orig = tools.search_emails
    tools.search_emails = fake_search_emails
    try:
        searches = [{"query": "alpha"}, {"query": "boom"}, {"query": "gamma"}]
        budget_ms = float(baseline.get("batch", {}).get("budget_ms", 5000))
        t0 = time.perf_counter()
        out = await tools.search_emails_batch(searches, user_id=H.test_user_id())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        tools.search_emails = orig

    # Contract: dict with one result entry per input, never raises.
    assert isinstance(out, dict), f"batch must return a dict, got {type(out)}"
    assert "results" in out, f"batch result missing 'results': {out}"
    assert len(out["results"]) == len(searches), "every input must yield a result entry"
    # The failing item is isolated as a per-item error, not a batch-level raise.
    boom = next(r for r in out["results"] if r["input"]["query"] == "boom")
    assert "error" in boom["result"], "failing item should carry a per-item error payload"
    good = [r for r in out["results"] if r["input"]["query"] != "boom"]
    assert all("results" in r["result"] for r in good), "non-failing items should carry results"
    ok, msg = H.check_regression("batch_wall", float(baseline.get("batch", {}).get("wall_ms", 50.0)), elapsed_ms)
    # Wall budget is a soft ceiling here (hermetic), but still gated.
    assert elapsed_ms <= budget_ms, f"batch exceeded wall budget {budget_ms}ms: {elapsed_ms:.1f}ms"
    print(msg)


# ── 2. End-to-end HTTP latency (SLOW / OPT-IN) ──────────────────────────────


@pytest.mark.perf_slow
def test_end_to_end_http_latency(serve_env, baseline, update_baseline):
    """SLOW GATE: drive the live /api/search over HTTP across representative
    queries; fail if total latency regresses past tolerance."""
    import httpx

    base_url = H.http_base_url(serve_env)
    if not H.serve_http_ready(base_url):
        pytest.skip(f"gmail-search serve not reachable at {base_url} — start it to run HTTP latency gate")

    token = serve_env.get("GMAIL_MCP_ADMIN_TOKEN")
    headers = {"X-User-Id": H.test_user_id()}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    measured: dict[str, float] = {}
    failures: list[str] = []

    for spec in REPRESENTATIVE_QUERIES:
        params = {"q": spec["q"], "k": spec["kwargs"].get("top_k", 10)}
        if spec["kwargs"].get("date_from"):
            params["date_from"] = spec["kwargs"]["date_from"]
        if spec["kwargs"].get("date_to"):
            params["date_to"] = spec["kwargs"]["date_to"]
        # Warm once (populate query_cache) then measure, mirroring real warm traffic.
        with httpx.Client(timeout=90) as client:
            client.get(f"{base_url}/api/search", params=params, headers=headers)
            t0 = time.perf_counter()
            resp = client.get(f"{base_url}/api/search", params=params, headers=headers)
            dt_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, f"{spec['name']}: HTTP {resp.status_code}: {resp.text[:200]}"
        measured[spec["name"]] = dt_ms

    if update_baseline:
        data = dict(baseline)
        data["end_to_end_http_ms"] = {k: round(v, 1) for k, v in measured.items()}
        H.save_baseline(data)
        pytest.skip("baseline updated (end_to_end_http_ms) — re-run without --update-baseline to gate")

    base = baseline.get("end_to_end_http_ms", {})
    for name, val in measured.items():
        if name not in base:
            failures.append(f"{name}: no baseline entry (run --update-baseline)")
            continue
        ok, msg = H.check_regression(f"e2e[{name}]", float(base[name]), val)
        print(msg)
        if not ok:
            failures.append(msg)
    assert not failures, "End-to-end latency regression:\n" + "\n".join(failures)


# ── 1. Per-stage timing (SLOW / OPT-IN — builds one engine) ─────────────────


@pytest.fixture(scope="module")
def engine():
    """Build ONE SearchEngine (loads ~10GB ScaNN index, ~100s). Module-scoped
    so all per-stage cases share it. Auto-skips without PG/corpus."""
    _require_corpus()
    from gmail_search.config import load_config
    from gmail_search.index.searcher import resolve_active_index_dir
    from gmail_search.search.engine import SearchEngine

    cfg = load_config(data_dir=H.REPO_DATA_DIR)
    # Resolve the SAME per-user index the live server uses (DB pointer ->
    # per-user fallback dir), instead of guessing a path.
    fallback = H.REPO_DATA_DIR / "users" / H.test_user_id() / "scann_index"
    index_dir = resolve_active_index_dir(H.REPO_DATA_DIR, fallback, user_id=H.test_user_id())
    try:
        eng = SearchEngine(
            db_path=H.REPO_DATA_DIR,
            index_dir=index_dir,
            config=cfg,
            user_id=H.test_user_id(),
        )
    except FileNotFoundError as e:
        pytest.skip(f"No built ScaNN index for {H.test_user_id()}: {e}")
    yield eng
    eng.close()


@pytest.mark.perf_slow
def test_per_stage_timing(engine, baseline, update_baseline):
    """SLOW GATE: time each stage of search_threads via monkeypatch (no
    engine.py edits) and fail if any stage regresses past tolerance.

    Stages: embed_query, resolve_candidate_msg_ids, scann_search,
    fetch_embedding_rows, bm25_search_fts, thread_summary_fetch,
    scoring_loop_residual, collapse_repeat_senders, llm_rerank,
    filter_offtopic, total. (Stages that don't fire for a given query —
    e.g. llm_rerank on a clear winner — are simply absent.)
    """
    # Warm the query cache so embed is the cached path (the realistic case).
    engine.search_threads("draw request", top_k=10)

    with H.StageTimer(engine) as timer:
        _results, stages, total_ms = timer.time_search("draw request", top_k=10)

    print("\nPer-stage timings (ms):")
    for k in sorted(stages):
        print(f"  {k:28s} {stages[k]:8.1f}")

    if update_baseline:
        data = dict(baseline)
        data["per_stage_ms"] = {k: round(v, 1) for k, v in stages.items()}
        H.save_baseline(data)
        pytest.skip("baseline updated (per_stage_ms) — re-run without --update-baseline to gate")

    base = baseline.get("per_stage_ms", {})
    failures: list[str] = []
    for name, val in stages.items():
        if name not in base:
            # New/optional stage with no baseline — informational, not a fail.
            print(f"  (no baseline for stage {name}={val:.1f}ms)")
            continue
        ok, msg = H.check_regression(f"stage[{name}]", float(base[name]), val)
        print(msg)
        if not ok:
            failures.append(msg)
    assert not failures, "Per-stage regression:\n" + "\n".join(failures)
