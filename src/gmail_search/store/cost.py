from datetime import datetime, timezone

TEXT_COST_PER_MILLION_TOKENS = 0.20
IMAGE_COST_PER_IMAGE = 0.0001


def estimate_cost(input_tokens: int = 0, image_count: int = 0) -> float:
    text_cost = (input_tokens / 1_000_000) * TEXT_COST_PER_MILLION_TOKENS
    image_cost = image_count * IMAGE_COST_PER_IMAGE
    return text_cost + image_cost


def record_cost(
    conn,
    operation: str,
    model: str,
    input_tokens: int,
    image_count: int,
    estimated_cost_usd: float,
    message_id: str,
    output_tokens: int = 0,
) -> None:
    """Append one row to the shared `costs` table.

    `output_tokens` defaults to 0 so existing callers (embed pipeline,
    chat summarizer) keep working unchanged — they don't have output
    tokens to record. Deep-analysis agents pass the real count so
    analytics can split input vs output without overloading the
    `image_count` column (which means "images processed" elsewhere).
    """
    conn.execute(
        """INSERT INTO costs (timestamp, operation, model, input_tokens,
           image_count, output_tokens, estimated_cost_usd, message_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            datetime.now(timezone.utc).isoformat(),
            operation,
            model,
            input_tokens,
            image_count,
            output_tokens,
            estimated_cost_usd,
            message_id,
        ),
    )
    conn.commit()


def get_total_spend(conn) -> float:
    row = conn.execute("SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM costs").fetchone()
    return row[0]


def get_spend_breakdown(conn) -> dict[str, float]:
    rows = conn.execute("SELECT operation, SUM(estimated_cost_usd) as total FROM costs GROUP BY operation").fetchall()
    return {r["operation"]: r["total"] for r in rows}


def check_budget(conn, max_budget_usd: float) -> tuple[bool, float, float]:
    spent = get_total_spend(conn)
    remaining = max_budget_usd - spent
    return remaining >= 0, spent, remaining
