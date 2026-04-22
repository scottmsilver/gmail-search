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


# ─── _extract_primary_links ────────────────────────────────────────────────
#
# The cleaned body has all URLs replaced by "[link]", which erases
# the one piece of information the reader usually wants for a
# shared-link email. We extract URLs from the raw body before
# cleaning and pass them to the LLM via `Links:` so the summary
# can cite the destination verbatim.


def test_extract_primary_links_bare_share():
    """Short body with a single share URL — Bruce's garage-sale case.

    Expected: that URL comes back as the primary link.
    """
    from gmail_search.summarize import _extract_primary_links

    body = (
        "Time For A Garage Sale: Collector's Items That Aren't Worth Keeping\n"
        "https://share.google/oNFNhOL9J2E6bIC0P\n\n"
        "Oh well!\n\n"
        "Bruce G. Silver, M.D.\nbrucesilvermd@gmail.com\n"
    )
    out = _extract_primary_links(body)
    assert out == ["https://share.google/oNFNhOL9J2E6bIC0P"]


def test_extract_primary_links_body_is_just_url():
    """Joy's tuna-recipe case — body is only the URL."""
    from gmail_search.summarize import _extract_primary_links

    body = "https://cooking.nytimes.com/recipes/1024664-crispy-tuna-cakes\n"
    out = _extract_primary_links(body)
    assert out == ["https://cooking.nytimes.com/recipes/1024664-crispy-tuna-cakes"]


def test_extract_primary_links_drops_unsubscribe_and_images():
    """Newsletters pack their bodies with unsubscribe links, pixel
    trackers, and inline-image URLs. None of those should win over a
    real article link.
    """
    from gmail_search.summarize import _extract_primary_links

    body = (
        "View in browser: https://mail.stratechery.com/view/abc123\n"
        "![hero](https://img.stratechery.com/hero.jpg)\n"
        "Read the article: https://stratechery.com/2026/the-real-bundle/\n"
        "\n---\n"
        "You're receiving this because you subscribe. "
        "Unsubscribe: https://mail.stratechery.com/unsubscribe?x=y\n"
        "Manage preferences: https://mail.stratechery.com/preferences\n"
    )
    out = _extract_primary_links(body)
    assert "https://stratechery.com/2026/the-real-bundle/" in out
    # the .jpg hero, unsub, preferences, view-in-browser must not appear
    assert all("unsubscribe" not in u.lower() for u in out)
    assert all("preferences" not in u.lower() for u in out)
    assert all(not u.endswith(".jpg") for u in out)


def test_extract_primary_links_dedupes_and_caps_at_two():
    from gmail_search.summarize import _extract_primary_links

    body = (
        "First: https://a.example.com/post\n"
        "Again: https://a.example.com/post\n"
        "Second: https://b.example.com/thing\n"
        "Third: https://c.example.com/other\n"
    )
    out = _extract_primary_links(body)
    assert out == [
        "https://a.example.com/post",
        "https://b.example.com/thing",
    ]


def test_extract_primary_links_empty_body():
    from gmail_search.summarize import _extract_primary_links

    assert _extract_primary_links("") == []
    assert _extract_primary_links("no urls here, just prose.") == []


def test_build_user_prompt_includes_links_line():
    """Extracted URLs should land in the prompt so the LLM can cite
    them verbatim — the prompt's FORWARDED/SHARED rule assumes a
    `Links:` block is available.
    """
    from gmail_search.summarize import _build_user_prompt

    body = "Check this out: https://example.com/article\n"
    prompt = _build_user_prompt("alice@example.com", "fwd:", body)
    assert "Links:" in prompt
    assert "https://example.com/article" in prompt


def test_build_user_prompt_no_links_line_when_no_urls():
    from gmail_search.summarize import _build_user_prompt

    prompt = _build_user_prompt("alice@example.com", "hi", "how are you")
    assert "Links:" not in prompt


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


def test_repair_broken_markdown_links():
    """The LLM occasionally closes a markdown link with `]` instead of
    `)` or runs out of tokens before emitting the close paren. Both
    cause the link to render as literal text (a long ugly URL). We
    repair on the way in.
    """
    from gmail_search.summarize import _repair_broken_markdown_links

    # Case 1: closing `]` instead of `)` — seen in the wild on msg
    # 19dac0e4a98f7854 (long tracking URL).
    broken = "Kohler reports generator started. [Generator Status](http://url/ls/click?x=y]"
    fixed = _repair_broken_markdown_links(broken)
    assert fixed == "Kohler reports generator started. [Generator Status](http://url/ls/click?x=y)"

    # Case 2: output truncated mid-link — no closing paren at all.
    # Drop the link rather than leaving dangling markdown in the UI.
    truncated = "Pay your invoice. [Pay now](https://pay.example.com/?token=abc"
    fixed = _repair_broken_markdown_links(truncated)
    # Link should close or be dropped — in both cases the result must
    # not contain a dangling `[Pay now](https://...` pattern.
    assert "[Pay now](" not in fixed or fixed.endswith(")")

    # Well-formed links pass through unchanged.
    good = "See [the article](https://example.com/post) for details."
    assert _repair_broken_markdown_links(good) == good

    # Multiple links, one broken.
    mixed = "See [a](https://a.com) and [b](https://b.com/?x=y]"
    fixed = _repair_broken_markdown_links(mixed)
    assert "[b](https://b.com/?x=y)" in fixed
    assert "[a](https://a.com)" in fixed


def test_record_and_clear_summary_failure(db_backend):
    """Failures land in `summary_failures`; the next success clears them.

    Drives the helpers directly — wiring into `_persist` is covered
    by the integration behaviour of the daemon at runtime.
    """
    from gmail_search.store.db import get_connection, init_db
    from gmail_search.summarize import _clear_summary_failure, _record_summary_failure

    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    # summary_failures has a FK to messages; stub one row.
    conn.execute(
        """INSERT INTO messages (id, thread_id, from_addr, to_addr, subject,
                                 body_text, date, labels)
           VALUES ('m1', 't1', 'a@x.com', 'b@x.com', 's', 'b',
                   '2026-04-21T00:00:00+00:00', '[]')"""
    )
    conn.commit()

    _record_summary_failure(conn, "m1", "fake-model+v5", "first error here")
    conn.commit()
    row = conn.execute("SELECT * FROM summary_failures WHERE message_id = 'm1'").fetchone()
    assert row["attempts"] == 1
    assert "first error" in row["error"]

    # Re-record: attempts increments, error updates.
    _record_summary_failure(conn, "m1", "fake-model+v5", "second error")
    conn.commit()
    row = conn.execute("SELECT * FROM summary_failures WHERE message_id = 'm1'").fetchone()
    assert row["attempts"] == 2
    assert "second" in row["error"]

    # A success clears the row.
    _clear_summary_failure(conn, "m1")
    conn.commit()
    row = conn.execute("SELECT * FROM summary_failures WHERE message_id = 'm1'").fetchone()
    assert row is None

    conn.close()


def test_batch_with_empty_input():
    from gmail_search.summarize import summarize_batch

    backend = FakeBackend(lambda _msgs: "{}")
    with _unused_client() as client:
        out = summarize_batch(client, [], backend)
    assert out == {}
