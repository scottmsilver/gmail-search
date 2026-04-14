from gmail_search.store.cost import check_budget, estimate_cost, get_spend_breakdown, get_total_spend, record_cost
from gmail_search.store.db import get_connection, init_db


def test_record_and_get_total_spend(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    record_cost(
        conn,
        operation="embed_text",
        model="test",
        input_tokens=1000,
        image_count=0,
        estimated_cost_usd=0.50,
        message_id="msg1",
    )
    record_cost(
        conn,
        operation="embed_text",
        model="test",
        input_tokens=2000,
        image_count=0,
        estimated_cost_usd=0.75,
        message_id="msg2",
    )
    total = get_total_spend(conn)
    assert abs(total - 1.25) < 0.001
    conn.close()


def test_get_spend_breakdown(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    record_cost(
        conn,
        operation="embed_text",
        model="test",
        input_tokens=1000,
        image_count=0,
        estimated_cost_usd=0.50,
        message_id="msg1",
    )
    record_cost(
        conn,
        operation="embed_image",
        model="test",
        input_tokens=0,
        image_count=5,
        estimated_cost_usd=0.10,
        message_id="msg1",
    )
    breakdown = get_spend_breakdown(conn)
    assert breakdown["embed_text"] == 0.50
    assert breakdown["embed_image"] == 0.10
    conn.close()


def test_check_budget_under(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    record_cost(
        conn,
        operation="embed_text",
        model="test",
        input_tokens=1000,
        image_count=0,
        estimated_cost_usd=1.00,
        message_id="msg1",
    )
    ok, spent, remaining = check_budget(conn, max_budget_usd=5.00)
    assert ok is True
    assert abs(spent - 1.00) < 0.001
    assert abs(remaining - 4.00) < 0.001
    conn.close()


def test_check_budget_over(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    record_cost(
        conn,
        operation="embed_text",
        model="test",
        input_tokens=1000,
        image_count=0,
        estimated_cost_usd=5.50,
        message_id="msg1",
    )
    ok, spent, remaining = check_budget(conn, max_budget_usd=5.00)
    assert ok is False
    assert remaining < 0
    conn.close()


def test_estimate_cost_text():
    cost = estimate_cost(input_tokens=1_000_000, image_count=0)
    assert abs(cost - 0.20) < 0.001


def test_estimate_cost_images():
    cost = estimate_cost(input_tokens=0, image_count=1000)
    assert abs(cost - 0.10) < 0.001


def test_estimate_cost_mixed():
    cost = estimate_cost(input_tokens=500_000, image_count=500)
    expected = 0.10 + 0.05
    assert abs(cost - expected) < 0.001
