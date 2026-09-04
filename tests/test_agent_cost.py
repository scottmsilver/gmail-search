"""Tests for the deep-analysis cost accounting helpers.

Verifies pricing-table matching (exact, prefix, default fallback),
zero-token quick-path, and the DB round-trip that lands rows in
the shared `costs` table with the expected operation tag.
"""

from __future__ import annotations

from gmail_search.agents.cost import GEMINI_PRICING, _match_pricing, estimate_agent_cost_usd, record_agent_cost
from gmail_search.agents.session import create_session, new_session_id
from gmail_search.store.db import get_connection, init_db


def test_pricing_exact_match():
    """Models that appear in the table get their own rates, not the
    fallback."""
    p = _match_pricing("gemini-2.5-pro")
    assert p is GEMINI_PRICING["gemini-2.5-pro"]
    assert p.input == 1.25
    assert p.output == 10.00


def test_pricing_prefix_match_longest_wins():
    """A flash-lite preview id must land on flash-lite pricing, not
    flash pricing (flash is a shorter prefix of the same string)."""
    p = _match_pricing("gemini-2.5-flash-lite-preview-01-01")
    assert p is GEMINI_PRICING["gemini-2.5-flash-lite"]


def test_pricing_unknown_model_falls_back_to_default():
    """A fresh model id falls back to the default row (flash-tier
    pricing) rather than crashing or recording 0.0 — under-counting
    silently is worse than slightly overestimating."""
    p = _match_pricing("gemini-99.9-magic")
    assert p is GEMINI_PRICING["default"]


def test_estimate_zero_tokens_returns_zero():
    """A degenerate call (connection dropped, safety refusal) with
    no token usage must not synthesize a phantom cost."""
    assert estimate_agent_cost_usd("gemini-2.5-pro", 0, 0) == 0.0


def test_estimate_respects_input_and_output_rates():
    """Sanity-check the math: 1M input + 1M output on pro should be
    $1.25 + $10.00 = $11.25."""
    assert estimate_agent_cost_usd("gemini-2.5-pro", 1_000_000, 1_000_000) == 11.25


def test_record_agent_cost_writes_deep_operation_row(db_backend):
    """The DB round-trip lands a row with operation `deep_<name>`
    and the estimated USD amount. Spend breakdown should segment
    per stage because that's the whole point."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    usd = record_agent_cost(
        conn,
        session_id=sid,
        agent_name="planner",
        model="gemini-2.5-flash",
        input_tokens=500,
        output_tokens=200,
    )
    assert usd > 0

    row = conn.execute(
        """SELECT operation, model, input_tokens, image_count, output_tokens,
                  estimated_cost_usd, message_id
             FROM costs WHERE message_id = %s""",
        (f"deep:{sid}",),
    ).fetchone()
    assert row["operation"] == "deep_planner"
    assert row["model"] == "gemini-2.5-flash"
    assert row["input_tokens"] == 500
    # Output tokens land in the dedicated `output_tokens` column;
    # `image_count` stays 0 because deep-mode never produces images.
    assert row["output_tokens"] == 200
    assert row["image_count"] == 0
    # Cost math: 500/1M * $0.075 + 200/1M * $0.30 ≈ $0.0000975
    assert row["estimated_cost_usd"] > 0
    conn.close()


def test_record_cost_writes_output_tokens_column(db_backend):
    """The shared `record_cost` writer accepts an `output_tokens`
    kwarg (default 0 for back-compat) and stores it in the dedicated
    column — without overloading `image_count`."""
    from gmail_search.store.cost import record_cost

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    record_cost(
        conn,
        operation="some_llm_op",
        model="gemini-2.5-flash",
        input_tokens=1234,
        image_count=0,
        output_tokens=567,
        estimated_cost_usd=0.01,
        message_id="msg-out-tokens",
    )

    row = conn.execute(
        """SELECT input_tokens, image_count, output_tokens
             FROM costs WHERE message_id = %s""",
        ("msg-out-tokens",),
    ).fetchone()
    assert row["input_tokens"] == 1234
    assert row["image_count"] == 0
    assert row["output_tokens"] == 567
    conn.close()


def test_record_agent_cost_participates_in_spend_breakdown(db_backend):
    """get_spend_breakdown() sums by operation; deep-mode rows should
    show up as their own `deep_<agent>` buckets, not hidden inside
    the generic 'embed_query' / 'summarize' lines."""
    from gmail_search.store.cost import get_spend_breakdown

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    record_agent_cost(
        conn,
        session_id=sid,
        agent_name="writer",
        model="gemini-2.5-pro",
        input_tokens=2000,
        output_tokens=500,
    )
    record_agent_cost(
        conn,
        session_id=sid,
        agent_name="planner",
        model="gemini-2.5-flash",
        input_tokens=300,
        output_tokens=100,
    )

    breakdown = get_spend_breakdown(conn)
    assert "deep_writer" in breakdown
    assert "deep_planner" in breakdown
    # Writer runs on pro → strictly more expensive per token than Planner on flash
    assert breakdown["deep_writer"] > breakdown["deep_planner"]
    conn.close()


def test_record_agent_cost_uses_override_when_given(monkeypatch):
    from gmail_search.agents import cost as cost_mod

    captured = {}

    def fake_record_cost(conn, **kw):
        captured.update(kw)

    monkeypatch.setattr(cost_mod, "record_cost", fake_record_cost)
    usd = cost_mod.record_agent_cost(
        object(),
        session_id="s1",
        agent_name="pi",
        model="anthropic/x",
        input_tokens=1000,
        output_tokens=10,
        usd_override=0.42,
    )
    assert usd == 0.42
    assert captured["estimated_cost_usd"] == 0.42
    assert captured["operation"] == "deep_pi"


def test_record_agent_cost_zero_override_not_treated_as_falsy(monkeypatch):
    from gmail_search.agents import cost as cost_mod

    captured = {}

    def fake_record_cost(conn, **kw):
        captured.update(kw)

    monkeypatch.setattr(cost_mod, "record_cost", fake_record_cost)
    usd = cost_mod.record_agent_cost(
        object(),
        session_id="s1",
        agent_name="test",
        model="gemini-2.5-flash",
        input_tokens=1_000_000,
        output_tokens=0,
        usd_override=0.0,
    )
    assert usd == 0.0
    assert captured["estimated_cost_usd"] == 0.0


def test_record_agent_cost_estimates_without_override(monkeypatch):
    from gmail_search.agents import cost as cost_mod

    monkeypatch.setattr(cost_mod, "record_cost", lambda conn, **kw: None)
    usd = cost_mod.record_agent_cost(
        object(), session_id="s1", agent_name="x", model="gemini-2.5-flash", input_tokens=1_000_000, output_tokens=0
    )
    assert usd == 0.075


def test_gemini_3_7_flash_pricing_is_not_the_default():
    from gmail_search.agents.cost import estimate_agent_cost_usd

    assert estimate_agent_cost_usd("gemini-3.7-flash", 1_000_000, 0) == 0.75


# ─── context-cache accounting (added 2026-09-04) ───────────────────────
#
# The pi runtime always reported cache_read_tokens, but service.py dropped
# it before record_agent_cost, so the ledger stored neither the split nor
# anything that could reproduce the invoice. These pin the semantics:
# input_tokens is FRESH only, cache reads are disjoint and bill ~10x lower.


def test_estimate_prices_cached_tokens_at_the_cached_rate():
    """Cached input is billed at `cached_input`, not `input`. Reproduces a
    real turn from 2026-09-03: 759,360 fresh + 3,483,568 cached + 26,845
    output on gemini-3.7-flash billed $0.93146."""
    usd = estimate_agent_cost_usd("gemini-3.7-flash", 759_360, 26_845, 3_483_568)
    assert round(usd, 5) == 0.93146


def test_estimate_cached_tokens_are_disjoint_from_input():
    """Passing tokens as cached must cost strictly less than passing the
    same tokens as fresh — the whole point of recording the split."""
    fresh_only = estimate_agent_cost_usd("gemini-3.7-flash", 1_000_000, 0)
    as_cached = estimate_agent_cost_usd("gemini-3.7-flash", 0, 0, 1_000_000)
    assert fresh_only == 0.75
    assert as_cached == 0.075
    assert estimate_agent_cost_usd("gemini-3.7-flash", 1_000_000, 0, 1_000_000) == 0.825


def test_estimate_falls_back_to_input_rate_without_a_cached_rate():
    """A model with no cached_input rate must overestimate (fresh rate)
    rather than silently price cache reads at zero."""
    assert GEMINI_PRICING["gemini-2.5-flash"].cached_input is None
    assert estimate_agent_cost_usd("gemini-2.5-flash", 0, 0, 1_000_000) == 0.075


def test_estimate_cached_only_call_is_not_zeroed():
    """A turn that was entirely cache hits still costs money; the
    zero-token quick path must not swallow it."""
    assert estimate_agent_cost_usd("gemini-3.7-flash", 0, 0, 500_000) > 0


def test_record_agent_cost_persists_the_cache_split(db_backend):
    """The counts land in their own columns, disjoint from input_tokens."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        session_id = new_session_id()
        create_session(conn, session_id=session_id, conversation_id=None, mode="deep", question="q")
        record_agent_cost(
            conn,
            session_id=session_id,
            agent_name="pi",
            model="google/gemini-3.7-flash",
            input_tokens=759_360,
            output_tokens=26_845,
            usd_override=0.93146,
            cache_read_tokens=3_483_568,
            cache_write_tokens=17,
        )
        row = conn.execute(
            "SELECT input_tokens, output_tokens, cached_input_tokens, cache_write_tokens,"
            " estimated_cost_usd FROM costs WHERE operation = 'deep_pi' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["input_tokens"] == 759_360
        assert row["cached_input_tokens"] == 3_483_568
        assert row["cache_write_tokens"] == 17
        assert round(row["estimated_cost_usd"], 5) == 0.93146
    finally:
        conn.close()


def test_record_agent_cost_defaults_cache_columns_to_zero(db_backend):
    """Callers that never had cache counts (embed path, ADK agents) keep
    working and read back 0, not NULL."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        session_id = new_session_id()
        create_session(conn, session_id=session_id, conversation_id=None, mode="deep", question="q")
        record_agent_cost(
            conn,
            session_id=session_id,
            agent_name="planner",
            model="gemini-2.5-flash",
            input_tokens=100,
            output_tokens=10,
        )
        row = conn.execute(
            "SELECT cached_input_tokens, cache_write_tokens FROM costs"
            " WHERE operation = 'deep_planner' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["cached_input_tokens"] == 0
        assert row["cache_write_tokens"] == 0
    finally:
        conn.close()


def test_record_agent_cost_estimate_uses_cache_read_when_no_override(monkeypatch):
    """Without a provider figure, the stored estimate must use the cached
    rate for cache reads instead of ignoring them."""
    import gmail_search.agents.cost as cost_mod

    captured = {}

    def _fake_record_cost(conn, **kw):
        captured.update(kw)

    monkeypatch.setattr(cost_mod, "record_cost", _fake_record_cost)
    usd = cost_mod.record_agent_cost(
        None,
        session_id="s",
        agent_name="pi",
        model="gemini-3.7-flash",
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=1_000_000,
    )
    assert usd == 0.825
    assert captured["cached_input_tokens"] == 1_000_000
