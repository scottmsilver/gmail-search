from gmail_search.gmail.parser import parse_message

SAMPLE_API_RESPONSE = {
    "id": "18f1234abcd",
    "threadId": "18f1234abcd",
    "historyId": "999999",
    "labelIds": ["INBOX", "UNREAD"],
    "payload": {
        "headers": [
            {"name": "From", "value": "Alice <alice@example.com>"},
            {"name": "To", "value": "Bob <bob@example.com>"},
            {"name": "Subject", "value": "Meeting tomorrow"},
            {"name": "Date", "value": "Mon, 15 Jun 2025 10:30:00 -0700"},
        ],
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "text/plain",
                "body": {"data": "SGVsbG8gV29ybGQ="},
            },
            {
                "mimeType": "text/html",
                "body": {"data": "PHA-SGVsbG8gV29ybGQ8L3A-"},
            },
            {
                "mimeType": "application/pdf",
                "filename": "report.pdf",
                "body": {"attachmentId": "ANGjdJ8abc", "size": 50000},
            },
        ],
    },
}


def test_parse_message_basic():
    msg, att_metas = parse_message(SAMPLE_API_RESPONSE)
    assert msg.id == "18f1234abcd"
    assert msg.thread_id == "18f1234abcd"
    assert msg.from_addr == "Alice <alice@example.com>"
    assert msg.to_addr == "Bob <bob@example.com>"
    assert msg.subject == "Meeting tomorrow"
    assert "Hello World" in msg.body_text
    assert msg.labels == ["INBOX", "UNREAD"]
    assert msg.history_id == 999999


def test_parse_message_attachments():
    msg, att_metas = parse_message(SAMPLE_API_RESPONSE)
    assert len(att_metas) == 1
    assert att_metas[0]["filename"] == "report.pdf"
    assert att_metas[0]["mime_type"] == "application/pdf"
    assert att_metas[0]["attachment_id"] == "ANGjdJ8abc"
    assert att_metas[0]["size"] == 50000


SIMPLE_API_RESPONSE = {
    "id": "simple1",
    "threadId": "simple1",
    "historyId": "100",
    "labelIds": [],
    "payload": {
        "headers": [
            {"name": "From", "value": "test@test.com"},
            {"name": "To", "value": "me@test.com"},
            {"name": "Subject", "value": "Simple"},
            {"name": "Date", "value": "Tue, 1 Jan 2025 00:00:00 +0000"},
        ],
        "mimeType": "text/plain",
        "body": {"data": "SnVzdCB0ZXh0"},
    },
}


def test_parse_simple_message():
    msg, att_metas = parse_message(SIMPLE_API_RESPONSE)
    assert msg.id == "simple1"
    assert "Just text" in msg.body_text
    assert len(att_metas) == 0


# ─── date handling ─────────────────────────────────────────────────────────
# Regression: a phishing email came through with a malformed `Date:`
# header ("04-03-2026") — RFC 2822 parsing failed and the parser fell
# back to 1970-01-01, cluttering the UI with "Dec 31 1969" messages.
# Gmail always stamps its own `internalDate` (ms since epoch); we
# should prefer that so malformed senders can't corrupt our timeline.

from datetime import datetime, timezone


def _msg_with(internal_date: str | None, date_header: str | None) -> dict:
    headers = [{"name": "From", "value": "a@b"}, {"name": "Subject", "value": "x"}]
    if date_header is not None:
        headers.append({"name": "Date", "value": date_header})
    raw: dict = {
        "id": "m1",
        "threadId": "m1",
        "historyId": "1",
        "labelIds": [],
        "payload": {"headers": headers, "mimeType": "text/plain", "body": {"data": "YQ=="}},
    }
    if internal_date is not None:
        raw["internalDate"] = internal_date
    return raw


def test_internal_date_preferred_over_date_header():
    # internalDate = 2024-01-15 12:00:00 UTC in ms
    raw = _msg_with(internal_date="1705320000000", date_header="Mon, 1 Jun 2020 00:00:00 +0000")
    msg, _ = parse_message(raw)
    assert msg.date == datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_malformed_date_header_with_valid_internal_date():
    # The real regression: `Date:` = "04-03-2026" (non-RFC-2822).
    # internalDate = 2026-03-04 14:30:00 UTC.
    raw = _msg_with(internal_date="1772872200000", date_header="04-03-2026")
    msg, _ = parse_message(raw)
    assert msg.date.year == 2026
    assert msg.date.month == 3
    # Must NOT be epoch 0.
    assert msg.date.year > 2000


def test_both_missing_falls_back_to_epoch():
    # Worst case (observed in spoofed phishing mail): internalDate="0"
    # AND Date header unparseable. Fall back to epoch 0 — unchanged
    # behaviour, so the UI can filter/label these explicitly.
    raw = _msg_with(internal_date="0", date_header="04-03-2026")
    msg, _ = parse_message(raw)
    assert msg.date == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_missing_date_header_uses_internal_date():
    raw = _msg_with(internal_date="1705320000000", date_header=None)
    msg, _ = parse_message(raw)
    assert msg.date == datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


# ─── Received-header fallback ──────────────────────────────────────────────
# When Gmail's `internalDate` field is 0 (observed on spoofed phishing)
# AND the sender's `Date:` is malformed, Gmail still stamps a proper
# receipt time into its first `Received:` header. Format:
#     Received: by <host> with SMTP id <id>; <RFC2822 date>
# Parse the RFC2822 part after the last semicolon.


def _msg_with_received(internal_date: str, date_header: str | None, received_values: list[str]) -> dict:
    headers = [{"name": "From", "value": "a@b"}, {"name": "Subject", "value": "x"}]
    for rv in received_values:
        headers.append({"name": "Received", "value": rv})
    if date_header is not None:
        headers.append({"name": "Date", "value": date_header})
    return {
        "id": "m1",
        "threadId": "m1",
        "historyId": "1",
        "labelIds": [],
        "internalDate": internal_date,
        "payload": {"headers": headers, "mimeType": "text/plain", "body": {"data": "YQ=="}},
    }


def test_received_header_used_when_internal_and_date_header_bad():
    # Exact format observed on the real 1969 phishing row.
    raw = _msg_with_received(
        internal_date="0",
        date_header="04-03-2026",
        received_values=[
            "by 2002:a05:651c:10a6:b0:38d:d91a:eb3 with SMTP id k6csp544234ljn;"
            "        Sat, 4 Apr 2026 16:31:03 -0700 (PDT)",
            "from efianalytics.com (efianalytics.com. 216.244.76.116)",
        ],
    )
    msg, _ = parse_message(raw)
    # 2026-04-04 16:31:03 -0700 == 2026-04-04 23:31:03 UTC.
    assert msg.date == datetime(2026, 4, 4, 23, 31, 3, tzinfo=timezone.utc)


def test_received_header_first_parseable_wins():
    # Multiple Received hops; take the FIRST one that parses (most
    # recent hop = Gmail's own stamp).
    raw = _msg_with_received(
        internal_date="0",
        date_header=None,
        received_values=[
            "from nowhere (no semicolon here at all)",
            "by google; Fri, 3 Apr 2026 10:00:00 +0000",
            "by earlier; Thu, 2 Apr 2026 10:00:00 +0000",
        ],
    )
    msg, _ = parse_message(raw)
    assert msg.date == datetime(2026, 4, 3, 10, 0, 0, tzinfo=timezone.utc)


def test_all_three_sources_bad_falls_back_to_epoch():
    raw = _msg_with_received(
        internal_date="0",
        date_header="04-03-2026",
        received_values=["from somewhere (no date in this hop)"],
    )
    msg, _ = parse_message(raw)
    assert msg.date == datetime(1970, 1, 1, tzinfo=timezone.utc)
