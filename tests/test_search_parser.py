"""Tests for the Gmail-style query parser.

Covers the six operator families we support:
  from:    to:    subject:    has:attachment    after:/before:    newer_than:/older_than:

Dates are normalised to ISO YYYY-MM-DD so they drop straight into the
existing search engine's date_from/date_to pipeline. Unrecognised
tokens stay in the residual freetext that's sent to semantic search.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gmail_search.search.parser import parse_query

# ─── plain freetext ────────────────────────────────────────────────────────


def test_plain_query_has_no_filters():
    pq = parse_query("meeting notes")
    assert pq.text == "meeting notes"
    assert pq.from_filter is None
    assert pq.to_filter is None
    assert pq.subject_filter is None
    assert pq.has_attachment is None
    assert pq.date_from is None
    assert pq.date_to is None


def test_empty_query():
    pq = parse_query("")
    assert pq.text == ""


# ─── from: / to: / subject: ───────────────────────────────────────────────


def test_from_simple_email():
    pq = parse_query("from:alice@example.com")
    assert pq.from_filter == "alice@example.com"
    assert pq.text == ""


def test_from_case_insensitive_prefix():
    pq = parse_query("FROM:alice@example.com urgent")
    assert pq.from_filter == "alice@example.com"
    assert pq.text == "urgent"


def test_from_quoted_value_with_space():
    pq = parse_query('from:"Alice Smith" urgent')
    assert pq.from_filter == "Alice Smith"
    assert pq.text == "urgent"


def test_to_simple():
    pq = parse_query("to:bob@example.com meeting")
    assert pq.to_filter == "bob@example.com"
    assert pq.text == "meeting"


def test_subject_simple():
    pq = parse_query("subject:report")
    assert pq.subject_filter == "report"
    assert pq.text == ""


def test_subject_quoted_multiword():
    pq = parse_query('subject:"Q4 Report" alice')
    assert pq.subject_filter == "Q4 Report"
    assert pq.text == "alice"


# ─── has:attachment ────────────────────────────────────────────────────────


def test_has_attachment_sets_flag_and_strips_token():
    pq = parse_query("has:attachment")
    assert pq.has_attachment is True
    assert pq.text == ""


def test_has_attachment_with_freetext():
    pq = parse_query("invoice has:attachment from 2024")
    assert pq.has_attachment is True
    # The rest stays in text (minus the consumed operator).
    assert pq.text == "invoice from 2024"


def test_has_unknown_value_stays_in_text():
    # has:foo shouldn't be silently consumed — we only recognise attachment.
    pq = parse_query("has:foo urgent")
    assert pq.has_attachment is None
    assert "has:foo" in pq.text
    assert "urgent" in pq.text


# ─── absolute date operators ───────────────────────────────────────────────


def test_after_iso_date_sets_date_from():
    pq = parse_query("after:2024-01-15")
    assert pq.date_from == "2024-01-15"
    assert pq.date_to is None
    assert pq.text == ""


def test_before_iso_date_sets_date_to():
    pq = parse_query("before:2024-12-31 construction")
    assert pq.date_to == "2024-12-31"
    assert pq.text == "construction"


def test_after_natural_language_yesterday():
    pq = parse_query("after:yesterday")
    expected = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    assert pq.date_from == expected


def test_before_quoted_natural_language():
    pq = parse_query('before:"last monday" reports')
    # Just assert it parses to SOME iso date — exact day depends on when
    # the test runs, but the function must not throw or leave it as text.
    assert pq.date_to is not None
    assert len(pq.date_to) == 10  # YYYY-MM-DD
    assert pq.text == "reports"


def test_bad_date_stays_in_text():
    pq = parse_query("after:tomorrowish meeting")
    # Unparseable → we do NOT silently drop the token. Keep it visible
    # in the freetext so the user can see what happened.
    assert pq.date_from is None
    assert "after:tomorrowish" in pq.text
    assert "meeting" in pq.text


# ─── relative date operators ───────────────────────────────────────────────


def test_newer_than_days_sets_date_from():
    pq = parse_query("newer_than:7d")
    expected = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
    assert pq.date_from == expected
    assert pq.date_to is None


def test_newer_than_weeks():
    pq = parse_query("newer_than:2w")
    expected = (datetime.now(timezone.utc).date() - timedelta(weeks=2)).isoformat()
    assert pq.date_from == expected


def test_older_than_year_sets_date_to():
    pq = parse_query("older_than:1y")
    today = datetime.now(timezone.utc).date()
    # ~365 days back, tolerate off-by-one on Feb 29 leap-year quirks.
    parsed = datetime.fromisoformat(pq.date_to).date()
    assert abs((today - parsed).days - 365) <= 1


def test_older_than_months():
    pq = parse_query("older_than:3m")
    today = datetime.now(timezone.utc).date()
    parsed = datetime.fromisoformat(pq.date_to).date()
    # 90 days gives us a sturdy window for 3 months without calendar drama.
    assert 80 <= (today - parsed).days <= 100


# ─── multi-operator queries ────────────────────────────────────────────────


def test_multi_operator_query():
    pq = parse_query("from:alice after:2024-01-01 subject:review meeting notes")
    assert pq.from_filter == "alice"
    assert pq.date_from == "2024-01-01"
    assert pq.subject_filter == "review"
    assert pq.text == "meeting notes"


def test_all_operator_families_together():
    pq = parse_query(
        'from:alice to:"Bob Smith" subject:"Q4 plan" has:attachment ' "after:2024-01-01 before:2024-12-31 roadmap"
    )
    assert pq.from_filter == "alice"
    assert pq.to_filter == "Bob Smith"
    assert pq.subject_filter == "Q4 plan"
    assert pq.has_attachment is True
    assert pq.date_from == "2024-01-01"
    assert pq.date_to == "2024-12-31"
    assert pq.text == "roadmap"


def test_newer_than_and_older_than_both_set():
    """Combining both is unusual but well-defined: window = (now-older,
    now-newer). Newer sets date_from (lower bound); older sets date_to.
    """
    pq = parse_query("newer_than:30d older_than:7d")
    assert pq.date_from is not None
    assert pq.date_to is not None
    # date_from should be older (earlier) than date_to.
    assert pq.date_from < pq.date_to


# ─── robustness ────────────────────────────────────────────────────────────


def test_unknown_prefix_stays_in_text():
    pq = parse_query("color:red hello")
    assert pq.text == "color:red hello"


def test_colon_without_prefix_stays_in_text():
    pq = parse_query("url:http://example.com")
    # Not one of our operators — treat as freetext.
    assert "url:http://example.com" in pq.text
