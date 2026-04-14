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
