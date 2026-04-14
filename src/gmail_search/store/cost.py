import sqlite3
from datetime import datetime, timezone

TEXT_COST_PER_MILLION_TOKENS = 0.20
IMAGE_COST_PER_IMAGE = 0.0001


def estimate_cost(input_tokens: int = 0, image_count: int = 0) -> float:
    text_cost = (input_tokens / 1_000_000) * TEXT_COST_PER_MILLION_TOKENS
    image_cost = image_count * IMAGE_COST_PER_IMAGE
    return text_cost + image_cost


def record_cost(
    conn: sqlite3.Connection,
    operation: str,
    model: str,
    input_tokens: int,
    image_count: int,
    estimated_cost_usd: float,
    message_id: str,
) -> None:
    conn.execute(
        """INSERT INTO costs (timestamp, operation, model, input_tokens,
           image_count, estimated_cost_usd, message_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            operation,
            model,
            input_tokens,
            image_count,
            estimated_cost_usd,
            message_id,
        ),
    )
    conn.commit()


def get_total_spend(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM costs").fetchone()
    return row[0]


def get_spend_breakdown(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute("SELECT operation, SUM(estimated_cost_usd) as total FROM costs GROUP BY operation").fetchall()
    return {r["operation"]: r["total"] for r in rows}


def check_budget(conn: sqlite3.Connection, max_budget_usd: float) -> tuple[bool, float, float]:
    spent = get_total_spend(conn)
    remaining = max_budget_usd - spent
    return remaining >= 0, spent, remaining
