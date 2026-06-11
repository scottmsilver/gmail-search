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


def test_denylist_auth_and_signup_endpoints():
    # Auth/signup walls render a form, never indexable content — measured as
    # ~half the pages that wastefully reach the browser tier.
    denied = [
        "https://nextdoor.com/login/?next=/news_feed",
        "https://www.dropbox.com/login?cont=%2Fhome",
        "https://login.wrike.com/login/?redirectUrl=x",
        "https://site.com/signin",
        "https://site.com/sign-in?x=1",
        "https://site.com/sign_in",
        "https://acme.com/oauth/authorize?client_id=1",
        "https://acme.com/auth/callback",
        "https://robertreich.substack.com/subscribe?r=77g1",
    ]
    for u in denied:
        assert _is_denied(u) is True, u
    # Content paths that merely CONTAIN an auth word must NOT be denied.
    for u in [
        "https://blog.example.com/how-to-login-securely",
        "https://news.example.com/the-authentication-problem",
        "https://shop.example.com/account-based-marketing-guide",
    ]:
        assert _is_denied(u) is False, u


def test_denylist_js_only_media_hosts():
    # Video / map / meeting links yield no crawlable text — whole-domain deny.
    for u in [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
        "https://meet.google.com/xyz-defg-hij",
        "https://maps.google.com/?q=somewhere",
    ]:
        assert _is_denied(u) is True, u
    # A same-name substring on a different domain must NOT be denied.
    assert _is_denied("https://notyoutube.example.com/article") is False


def test_unwrap_proofpoint_v2():
    from gmail_search.gmail.url_extract import unwrap_tracker_url
    url = "https://urldefense.proofpoint.com/v2/url?u=https-3A__proptia.odoo.com_knowledge_article_95&d=Dw&c=ab"
    assert unwrap_tracker_url(url) == "https://proptia.odoo.com/knowledge/article/95"


def test_unwrap_proofpoint_v3():
    from gmail_search.gmail.url_extract import unwrap_tracker_url
    url = "https://urldefense.com/v3/__https://twitter.com/ACScloudpartner__;!!CQl3mcHX2A!H7v$"
    assert unwrap_tracker_url(url) == "https://twitter.com/ACScloudpartner"


def test_unwrap_safelinks():
    from gmail_search.gmail.url_extract import unwrap_tracker_url
    url = "https://nam12.safelinks.protection.outlook.com/?url=https%3A%2F%2Freal.example.com%2Fdoc&data=x"
    assert unwrap_tracker_url(url) == "https://real.example.com/doc"


def test_unwrap_google_url():
    from gmail_search.gmail.url_extract import unwrap_tracker_url
    assert unwrap_tracker_url("https://www.google.com/url?q=https://dest.example.com/a&sa=D") == "https://dest.example.com/a"


def test_unwrap_passthrough_non_wrapper():
    from gmail_search.gmail.url_extract import unwrap_tracker_url
    # A normal URL (and an opaque tracker we CAN'T decode) is returned unchanged.
    assert unwrap_tracker_url("https://blog.example.com/post") == "https://blog.example.com/post"
    assert unwrap_tracker_url("https://track.worddaily.com/?xtl=abc123") == "https://track.worddaily.com/?xtl=abc123"


def test_unwrap_nested():
    from gmail_search.gmail.url_extract import unwrap_tracker_url
    # SafeLink wrapping a Google redirect → fully unwrapped.
    inner = "https%3A%2F%2Fwww.google.com%2Furl%3Fq%3Dhttps%3A%2F%2Ffinal.example.com%2Fx"
    url = f"https://safelinks.protection.outlook.com/?url={inner}"
    assert unwrap_tracker_url(url) == "https://final.example.com/x"


def test_extract_decodes_wrappers_and_dedups():
    from gmail_search.gmail.url_extract import extract_crawlable_urls
    # Two different proofpoint tokens wrapping the SAME destination dedup to one.
    body = (
        "https://urldefense.proofpoint.com/v2/url?u=https-3A__dest.example.com_a&d=Dw&c=1\n"
        "https://urldefense.proofpoint.com/v2/url?u=https-3A__dest.example.com_a&d=Xy&c=2\n"
    )
    out = extract_crawlable_urls(body)
    assert out == ["https://dest.example.com/a"]
