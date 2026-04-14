from gmail_search.embed.client import estimate_tokens, format_message_text, truncate_to_token_limit


def test_estimate_tokens():
    text = "Hello world this is a test"
    tokens = estimate_tokens(text)
    assert 4 <= tokens <= 10


def test_truncate_to_token_limit():
    short = "Hello world"
    assert truncate_to_token_limit(short, 8192) == short

    long_text = "word " * 10000
    truncated = truncate_to_token_limit(long_text, 100)
    tokens = estimate_tokens(truncated)
    assert tokens <= 110


def test_format_message_text():
    text = format_message_text(
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        date="2025-06-15",
        subject="Hello",
        body="Body text here",
    )
    assert "From: alice@example.com" in text
    assert "To: bob@example.com" in text
    assert "Subject: Hello" in text
    assert "Body text here" in text
