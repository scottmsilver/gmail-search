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
