"""Gmail-style query parser.

Extracts structured filters out of a raw query string:

    from:alice@example.com    to:"Bob Smith"    subject:"Q4 plan"
    has:attachment
    after:2024-01-15    before:"last monday"
    newer_than:7d       older_than:1y

Whatever tokens are not recognised fall through as residual freetext
which the engine feeds to semantic search.

Dates (absolute + relative) are normalised to ISO YYYY-MM-DD so they
drop straight into the existing search engine's date_from/date_to
pipeline without the engine learning about operator syntax.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured form of a raw Gmail-style query string."""

    text: str
    from_filter: Optional[str] = None
    to_filter: Optional[str] = None
    subject_filter: Optional[str] = None
    has_attachment: Optional[bool] = None  # None = no constraint
    date_from: Optional[str] = None  # ISO YYYY-MM-DD (inclusive)
    date_to: Optional[str] = None  # ISO YYYY-MM-DD (inclusive)
    # Soft signal: how strongly the original query implied "recent". Fed
    # to the engine as a recency-weight boost.
    temporal_boost: float = 0.0


# Operators that take a value after the colon.
_STRING_OPERATORS = {"from", "to", "subject"}
_DATE_OPERATORS_ABSOLUTE = {"after", "before"}
_DATE_OPERATORS_RELATIVE = {"newer_than", "older_than"}
_BOOL_OPERATORS = {"has"}  # only has:attachment is recognised

# e.g. "7d", "2w", "3m", "1y"
_RELATIVE_DURATION_RE = re.compile(r"^(\d+)\s*([dwmy])$", re.IGNORECASE)

# Temporal-intent signals in freetext (unchanged from the old parser — we
# keep the behaviour so recency boosts still fire on e.g. "latest notes").
_TEMPORAL_PATTERNS: list[tuple[str, float]] = [
    (r"\b(today|yesterday|this morning|tonight)\b", 0.35),
    (r"\b(this week|last week|few days ago)\b", 0.30),
    (r"\b(this month|last month|recently|recent)\b", 0.25),
    (r"\b(this year|last year|past year)\b", 0.15),
    (r"\b(latest|newest|most recent)\b", 0.30),
]


def _today_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_absolute_date(value: str) -> Optional[str]:
    """Parse an `after:`/`before:` value into ISO YYYY-MM-DD. Returns
    None if dateparser can't make sense of it even after common
    normalisations.
    """
    import dateparser  # inline — dateparser is slow to import

    settings = {"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True, "PREFER_DATES_FROM": "past"}
    dt = dateparser.parse(value, settings=settings)
    if dt is None:
        # dateparser handles bare weekday names ("monday" → most recent
        # Monday) but not "last monday". Strip the leading "last " and
        # retry — we still prefer past via settings, so the semantics
        # stay correct.
        stripped = re.sub(r"^\s*last\s+", "", value, flags=re.IGNORECASE)
        if stripped != value:
            dt = dateparser.parse(stripped, settings=settings)
    if dt is None:
        return None
    return dt.date().isoformat()


def _parse_relative_duration(value: str) -> Optional[timedelta]:
    """Parse an `Nd`/`Nw`/`Nm`/`Ny` value into a timedelta."""
    m = _RELATIVE_DURATION_RE.match(value.strip())
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "d":
        return timedelta(days=n)
    if unit == "w":
        return timedelta(weeks=n)
    if unit == "m":
        # Approximate a month as 30 days. Good enough for a filter window.
        return timedelta(days=30 * n)
    if unit == "y":
        return timedelta(days=365 * n)
    return None


def _split_prefix(token: str) -> Optional[tuple[str, str]]:
    """Return (prefix, value) if `token` looks like `prefix:value`, else
    None. Prefix is lowercased so `FROM:` and `from:` behave the same.
    """
    if ":" not in token:
        return None
    prefix, _, value = token.partition(":")
    if not prefix or not value:
        return None
    return prefix.lower(), value


def _compute_temporal_boost(text: str) -> float:
    boost = 0.0
    for pattern, b in _TEMPORAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            boost = max(boost, b)
    return boost


def _tokenise(raw: str) -> list[str]:
    """Split the query into tokens respecting double-quoted values so
    `subject:"Q4 plan"` stays together as one token `subject:Q4 plan`.
    Falls back to simple whitespace split if shlex chokes on unbalanced
    quotes.
    """
    try:
        return shlex.split(raw, posix=True)
    except ValueError:
        return raw.split()


def parse_query(raw: str) -> ParsedQuery:
    """Extract structured filters from a Gmail-style query string."""
    residual: list[str] = []
    from_filter: Optional[str] = None
    to_filter: Optional[str] = None
    subject_filter: Optional[str] = None
    has_attachment: Optional[bool] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    for token in _tokenise(raw):
        parts = _split_prefix(token)
        if parts is None:
            residual.append(token)
            continue

        prefix, value = parts

        if prefix in _STRING_OPERATORS:
            if prefix == "from":
                from_filter = value
            elif prefix == "to":
                to_filter = value
            else:
                subject_filter = value
            continue

        if prefix in _BOOL_OPERATORS:
            if prefix == "has" and value.lower() == "attachment":
                has_attachment = True
                continue
            # Unknown has:foo — leave it in freetext so the user notices.
            residual.append(token)
            continue

        if prefix in _DATE_OPERATORS_ABSOLUTE:
            iso = _parse_absolute_date(value)
            if iso is None:
                # Unparseable date — don't silently consume. Preserve the
                # original token so the user can spot the typo.
                residual.append(token)
                continue
            if prefix == "after":
                date_from = iso
            else:  # before
                date_to = iso
            continue

        if prefix in _DATE_OPERATORS_RELATIVE:
            delta = _parse_relative_duration(value)
            if delta is None:
                residual.append(token)
                continue
            target = (_today_utc() - delta).date().isoformat()
            if prefix == "newer_than":
                # "newer_than:7d" = messages from the last 7 days onward.
                date_from = target
            else:  # older_than
                date_to = target
            continue

        # Known-shape token (prefix:value) but unknown prefix — keep it
        # as-is in the freetext.
        residual.append(token)

    text = " ".join(residual).strip()
    return ParsedQuery(
        text=text,
        from_filter=from_filter,
        to_filter=to_filter,
        subject_filter=subject_filter,
        has_attachment=has_attachment,
        date_from=date_from,
        date_to=date_to,
        temporal_boost=_compute_temporal_boost(text),
    )
