"""URL extraction from email bodies — regex + denylist behaviour."""

from __future__ import annotations

from gmail_search.gmail.url_extract import _is_denied, extract_crawlable_urls


def test_extracts_plain_url():
    body = "Hey, check out https://example.com/foo for details."
    assert extract_crawlable_urls(body) == ["https://example.com/foo"]


def test_strips_trailing_punctuation():
    body = "See https://example.com/path, it's great. Also https://good.org!"
    urls = extract_crawlable_urls(body)
    assert "https://example.com/path" in urls
    assert "https://good.org" in urls
    # No punctuation stuck to either one.
    for url in urls:
        assert url[-1] not in ".,;:!?"


def test_dedups_preserving_order():
    body = (
        "First https://alpha.example.com/a\n"
        "Then https://beta.example.com\n"
        "And again https://alpha.example.com/a\n"
    )
    urls = extract_crawlable_urls(body)
    assert urls == ["https://alpha.example.com/a", "https://beta.example.com"]


def test_denylist_click_tracker_hosts():
    body = "https://click.mail.vendor.com/abc123\n" "https://ct.sendgrid.net/xyz\n" "https://trk.klclick1.com/abc\n"
    assert extract_crawlable_urls(body) == []


def test_denylist_unsubscribe_paths():
    body = (
        "https://news.example.com/unsubscribe?u=12345\n"
        "https://news.example.com/email-preferences\n"
        "https://fine.example.com/article/1"
    )
    urls = extract_crawlable_urls(body)
    assert "https://fine.example.com/article/1" in urls
    assert all("unsubscribe" not in u for u in urls)
    assert all("preferences" not in u for u in urls)


def test_denylist_binary_suffixes():
    body = (
        "https://cdn.example.com/logo.png\n"
        "https://cdn.example.com/report.pdf_hint\n"  # not a pdf suffix
        "https://cdn.example.com/clip.mp4\n"
        "https://cdn.example.com/event.ics\n"
        "https://site.example.com/article\n"
    )
    urls = extract_crawlable_urls(body)
    # Article passes; binary-suffixed ones dropped.
    assert "https://site.example.com/article" in urls
    assert not any(u.endswith(".png") for u in urls)
    assert not any(u.endswith(".mp4") for u in urls)
    assert not any(u.endswith(".ics") for u in urls)


def test_denylist_drive_and_gmail():
    body = (
        "https://docs.google.com/document/d/abc123/edit\n"
        "https://drive.google.com/file/d/xyz/view\n"
        "https://mail.google.com/mail/u/0/#inbox\n"
        "https://example.com/ok\n"
    )
    urls = extract_crawlable_urls(body)
    assert urls == ["https://example.com/ok"]


def test_denylist_social_profiles():
    body = (
        "https://linkedin.com/in/someone\n"
        "https://twitter.com/user/status/123\n"
        "https://x.com/user\n"
        "https://facebook.com/pageish\n"
    )
    assert extract_crawlable_urls(body) == []


def test_parens_and_quotes_not_captured():
    body = 'Quote: "https://example.com/x" and paren (https://example.com/y).'
    urls = extract_crawlable_urls(body)
    assert "https://example.com/x" in urls
    assert "https://example.com/y" in urls


def test_empty_body():
    assert extract_crawlable_urls("") == []
    assert extract_crawlable_urls(None) == []  # type: ignore[arg-type]


def test_http_scheme_ok():
    body = "Old link: http://legacy.example.com/page"
    assert extract_crawlable_urls(body) == ["http://legacy.example.com/page"]


def test_is_denied_helpers_direct():
    assert _is_denied("https://click.mail.vendor.com/a") is True
    assert _is_denied("https://news.com/manage-subscription") is True
    assert _is_denied("https://cdn.com/a.png") is True
    assert _is_denied("https://example.com/article") is False
    # Malformed / missing host.
    assert _is_denied("not-a-url") is True
