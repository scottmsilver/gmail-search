"""Cost accounting for the deep-analysis pipeline.

Each sub-agent invocation through `adk_invoke` runs one LLM call. ADK
events carry `usage_metadata` with prompt / candidates token counts;
this module turns those into dollar estimates and appends a row to
the existing `costs` table (same table the embed + chat paths write
to, so `get_total_spend` / `get_spend_breakdown` already aggregate
deep-mode alongside everything else).

Operation labels: we write `deep_<agent_name>` (e.g. `deep_planner`,
`deep_writer`) so the spend breakdown can tell per-stage costs apart.
That makes it easy to see if the Writer on gemini-2.5-pro is what's
eating the budget vs the Planner on flash.

Pricing constants are the published Gemini rates as of Jan 2026.
Tune if Google changes them — `GEMINI_PRICING` is the only place
numbers live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from gmail_search.store.cost import record_cost

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pricing:
    """Per-million-token rates in USD.

    `cached_input` is the rate for prompt tokens served from the
    provider-side context cache — roughly a tenth of `input` on the
    flash tier. It IS consulted now: `estimate_agent_cost_usd` takes a
    `cached_input_tokens` argument, and the pi runtime reports the split
    (`pi_protocol.UsageStats.cache_read_tokens`). Models with no
    `cached_input` rate fall back to the fresh `input` rate, which
    overestimates rather than hides the cost."""

    input: float
    output: float
    cached_input: float | None = None


# Rates as of Jan 2026. Updating this table is the one place to touch
# when Google changes prices.
GEMINI_PRICING: dict[str, Pricing] = {
    "gemini-2.5-pro": Pricing(input=1.25, output=10.00, cached_input=0.3125),
    "gemini-2.5-flash": Pricing(input=0.075, output=0.30),
    "gemini-2.5-flash-lite": Pricing(input=0.10, output=0.40),
    "gemini-3.1-pro-preview": Pricing(input=1.25, output=10.00),
    "gemini-3.7-flash": Pricing(input=0.75, output=3.75, cached_input=0.075),
    # Fallback; matches Flash pricing so we never undercount by
    # accident when a brand-new model id shows up.
    "default": Pricing(input=0.075, output=0.30),
}


def _match_pricing(model: str) -> Pricing:
    """Pick the pricing row that matches the model name. Exact match
    first, then prefix scan (so `gemini-2.5-flash-preview-03-25`
    still lands on gemini-2.5-flash), then fall back to default."""
    if not model:
        return GEMINI_PRICING["default"]
    if model in GEMINI_PRICING:
        return GEMINI_PRICING[model]
    # Longest prefix wins so `gemini-2.5-flash-lite` beats
    # `gemini-2.5-flash` when both prefix-match.
    candidates = sorted(
        (k for k in GEMINI_PRICING if k != "default" and model.startswith(k)),
        key=len,
        reverse=True,
    )
    if candidates:
        return GEMINI_PRICING[candidates[0]]
    logger.info(f"no pricing for model {model!r}; using default (flash-tier) rates")
    return GEMINI_PRICING["default"]


def estimate_agent_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Per-call USD estimate. Returns 0.0 on zero tokens so a
    degenerate call that streamed nothing doesn't synthesize a cost
    row with phantom prices.

    `input_tokens` is FRESH prompt tokens and `cached_input_tokens` is
    the context-cache read; the two are disjoint, matching what the pi
    runtime reports. Verified against a real turn on 2026-09-03:
    759,360 fresh + 3,483,568 cached + 26,845 output on
    gemini-3.7-flash gives $0.93146, the figure the provider billed.
    """
    if input_tokens <= 0 and output_tokens <= 0 and cached_input_tokens <= 0:
        return 0.0
    p = _match_pricing(model)
    cached_rate = p.cached_input if p.cached_input is not None else p.input
    return (
        (input_tokens / 1_000_000) * p.input
        + (cached_input_tokens / 1_000_000) * cached_rate
        + (output_tokens / 1_000_000) * p.output
    )


def record_agent_cost(
    conn,
    *,
    session_id: str,
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    usd_override: float | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Append one `deep_<agent_name>` row to the `costs` table and
    return the estimated USD amount. Reuses the shared `record_cost`
    writer so spend reporting (total, breakdown, budget check)
    picks this up automatically.

    Output tokens land in the dedicated `output_tokens` column (added
    to the `costs` table specifically so we don't have to overload
    `image_count`, which means "images processed" for the embed
    pipeline). `image_count` stays 0 for deep-mode rows since text-only
    LLM calls produce no images.

    `usd_override`, when given, is a provider-reported figure (pi's
    `get_session_stats.cost`) and replaces the pricing-table estimate,
    which only knows Gemini rates.

    `cache_read_tokens` / `cache_write_tokens` are the provider-cache
    counts, disjoint from `input_tokens`. They are stored in their own
    columns AND fed to the estimate, so a row can be re-priced later
    without re-running the agent. Before 2026-09-04 the pi runtime sent
    these and they were dropped here, which made the ledger impossible
    to reconcile against the invoice.
    """
    usd = (
        usd_override
        if usd_override is not None
        else estimate_agent_cost_usd(model, input_tokens, output_tokens, cache_read_tokens)
    )
    # session_id is threaded through `message_id` — the column is
    # TEXT + required, and a session id is as good an anchor as any
    # for deep-mode rows. Prefix with `deep:` to make it unambiguous
    # on the off-chance a real Gmail message id collides.
    record_cost(
        conn,
        operation=f"deep_{agent_name}",
        model=model,
        input_tokens=input_tokens,
        image_count=0,
        output_tokens=output_tokens,
        cached_input_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        estimated_cost_usd=usd,
        message_id=f"deep:{session_id}",
    )
    return usd
