from datetime import datetime, timezone
from typing import Optional

from gmail_search.auth.write_user import resolve_write_user_id

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
    *,
    user_id: Optional[str] = None,
) -> None:
    """Append one row to the per-user `costs` table.

    `output_tokens` defaults to 0 so existing callers (embed pipeline,
    chat summarizer) keep working unchanged — they don't have output
    tokens to record. Deep-analysis agents pass the real count so
    analytics can split input vs output without overloading the
    `image_count` column (which means "images processed" elsewhere).
    """
    uid = resolve_write_user_id(conn, user_id=user_id)
    conn.execute(
        """INSERT INTO costs (timestamp, operation, model, input_tokens,
           image_count, output_tokens, estimated_cost_usd, message_id, user_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            datetime.now(timezone.utc).isoformat(),
            operation,
            model,
            input_tokens,
            image_count,
            output_tokens,
            estimated_cost_usd,
            message_id,
            uid,
        ),
    )
    conn.commit()


def get_total_spend(conn, *, user_id: Optional[str] = None) -> float:
    if user_id is not None:
        row = conn.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM costs WHERE user_id = %s",
            (user_id,),
        ).fetchone()
    else:
        row = conn.execute("SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM costs").fetchone()
    return row[0]


def get_spend_breakdown(conn, *, user_id: Optional[str] = None) -> dict[str, float]:
    if user_id is not None:
        rows = conn.execute(
            "SELECT operation, SUM(estimated_cost_usd) as total FROM costs " "WHERE user_id = %s GROUP BY operation",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT operation, SUM(estimated_cost_usd) as total FROM costs GROUP BY operation"
        ).fetchall()
    return {r["operation"]: r["total"] for r in rows}


def check_budget(conn, max_budget_usd: float, *, user_id: Optional[str] = None) -> tuple[bool, float, float]:
    spent = get_total_spend(conn, user_id=user_id)
    remaining = max_budget_usd - spent
    return remaining >= 0, spent, remaining
