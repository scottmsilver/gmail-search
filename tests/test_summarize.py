"""Tests for the summarize module — auto-classify logic and the
multi-email batching path.

LLM transport is stubbed via a FakeBackend — no httpx, no Ollama, no
vLLM. The Backend abstraction (gmail_search.llm) is the right seam: we
plug in a class that returns canned responses and track the calls it
received.
"""

from __future__ import annotations

import json

import httpx


class FakeBackend:
    """Canned-response Backend for tests. `responder(messages)` receives
    the chat messages list and returns the raw content string.
    """

    model_id = "fake-model"

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[list[dict]] = []

    def chat(self, client, messages, *, max_tokens, json_format=False):
        self.calls.append(list(messages))
        return self.responder(messages)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


def _unused_client():
    """summarize_* takes an httpx.Client as an abstraction hook. Under
    test we never actually make HTTP calls (FakeBackend short-circuits
    everything) so a plain Client is fine.
    """
    return httpx.Client()


# ─── _auto_mail_summary ────────────────────────────────────────────────────


def test_auto_classify_promotions():
    from gmail_search.summarize import _auto_mail_summary

    out = _auto_mail_summary('["CATEGORY_PROMOTIONS", "INBOX"]', '"Acme Deals" <deals@acme.com>')
    assert out == "Promotional email from Acme Deals."


def test_auto_classify_social():
    from gmail_search.summarize import _auto_mail_summary

    out = _auto_mail_summary('["CATEGORY_SOCIAL"]', "Facebook <no-reply@facebookmail.com>")
    assert out and out.startswith("Social-network update from")


def test_auto_classify_updates_does_not_short_circuit():
    """CATEGORY_UPDATES is receipts/security codes — must reach the LLM
    so we extract the amount / code.
    """
    from gmail_search.summarize import _auto_mail_summary

    out = _auto_mail_summary('["CATEGORY_UPDATES"]', "OpenAI <noreply@openai.com>")
    assert out is None


def test_auto_classify_plain_personal_mail():
    from gmail_search.summarize import _auto_mail_summary

    out = _auto_mail_summary('["INBOX", "IMPORTANT"]', "Rebecca <rebecca@example.com>")
    assert out is None


def test_auto_classify_malformed_labels_json():
    from gmail_search.summarize import _auto_mail_summary

    out = _auto_mail_summary("not valid json", "someone@example.com")
    assert out is None  # fall through to LLM rather than crash


# ─── _clean_body ───────────────────────────────────────────────────────────


def test_clean_body_strips_urls():
    from gmail_search.summarize import _clean_body

    s = "See https://example.com/long/path?a=b and also http://foo.bar/baz for details"
    out = _clean_body(s)
    assert "http" not in out
    assert "[link]" in out


def test_clean_body_strips_invisible_preview_chars():
    from gmail_search.summarize import _clean_body

    # U+034F is the one newsletters use
    s = "Hello \u034f\u034f\u034f\u034f\u034f\u034f world"
    out = _clean_body(s)
    assert "\u034f" not in out


def test_clean_body_strips_mime_headers():
    from gmail_search.summarize import _clean_body

    s = "Content-Type: text/plain; charset=UTF-8\nContent-Transfer-Encoding: 7bit\n\nActual body."
    out = _clean_body(s)
    assert "Content-Type" not in out
    assert "Actual body" in out


# ─── summarize_batch ───────────────────────────────────────────────────────


def test_batch_summarizes_all_emails_when_json_parses():
    from gmail_search.summarize import summarize_batch

    messages = [
        {"id": "m1", "from_addr": "a@x.com", "subject": "s1", "body_text": "b1", "labels_json": "[]"},
        {"id": "m2", "from_addr": "b@x.com", "subject": "s2", "body_text": "b2", "labels_json": "[]"},
        {"id": "m3", "from_addr": "c@x.com", "subject": "s3", "body_text": "b3", "labels_json": "[]"},
    ]
    backend = FakeBackend(lambda _msgs: json.dumps({"m1": "one", "m2": "two", "m3": "three"}))

    with _unused_client() as client:
        out = summarize_batch(client, messages, backend)

    assert out == {"m1": "one", "m2": "two", "m3": "three"}


def test_batch_short_circuits_auto_classified_messages():
    """Marketing + social mail shouldn't be sent to the LLM even inside
    a batch. They get canonical summaries and the LLM call only gets
    the remaining messages.
    """
    from gmail_search.summarize import summarize_batch

    messages = [
        {
            "id": "mp",
            "from_addr": "Acme <deals@acme.com>",
            "subject": "Sale!",
            "body_text": "body",
            "labels_json": '["CATEGORY_PROMOTIONS"]',
        },
        {"id": "real", "from_addr": "alice@example.com", "subject": "lunch", "body_text": "noon?", "labels_json": "[]"},
    ]
    backend = FakeBackend(lambda _msgs: json.dumps({"real": "Alice asks about lunch at noon."}))

    with _unused_client() as client:
        out = summarize_batch(client, messages, backend)

    assert out["mp"].startswith("Promotional")
    assert "Alice" in out["real"]
    # Only one LLM call — `mp` bypassed it.
    assert len(backend.calls) == 1
    # That one call didn't include the promo message in its body.
    user_msg = next(m for m in backend.calls[0] if m["role"] == "user")
    assert "Acme" not in user_msg["content"]


def test_batch_falls_back_to_per_email_on_parse_failure():
    from gmail_search.summarize import summarize_batch

    messages = [
        {"id": "a", "from_addr": "x@x.com", "subject": "sA", "body_text": "bA", "labels_json": "[]"},
        {"id": "b", "from_addr": "y@y.com", "subject": "sB", "body_text": "bB", "labels_json": "[]"},
    ]
    # First call (batch) returns un-parseable text. Subsequent calls
    # (per-email fallback) return plain summary strings.
    responses = ["not json at all, just prose", "fallback summary for A", "fallback summary for B"]

    def responder(_msgs):
        return responses.pop(0)

    backend = FakeBackend(responder)
    with _unused_client() as client:
        out = summarize_batch(client, messages, backend)

    assert out == {"a": "fallback summary for A", "b": "fallback summary for B"}


def test_batch_fills_missing_ids_via_per_email_fallback():
    """If the batch JSON parses but one ID is missing, the missing one
    gets a per-email retry.
    """
    from gmail_search.summarize import summarize_batch

    messages = [
        {"id": "a", "from_addr": "x@x.com", "subject": "sA", "body_text": "bA", "labels_json": "[]"},
        {"id": "b", "from_addr": "y@y.com", "subject": "sB", "body_text": "bB", "labels_json": "[]"},
    ]
    responses = [json.dumps({"a": "batch summary A"}), "per-email summary B"]  # "b" missing
    backend = FakeBackend(lambda _msgs: responses.pop(0))

    with _unused_client() as client:
        out = summarize_batch(client, messages, backend)

    assert out["a"] == "batch summary A"
    assert out["b"] == "per-email summary B"


def test_batch_with_empty_input():
    from gmail_search.summarize import summarize_batch

    backend = FakeBackend(lambda _msgs: "{}")
    with _unused_client() as client:
        out = summarize_batch(client, [], backend)
    assert out == {}
