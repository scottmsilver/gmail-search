"""Extract crawlable URLs from email bodies.

Pure-Python: regex + denylist, no I/O. Callers (client.py at ingest time,
the CLI catch-up scan) feed each returned URL to `upsert_url_stub` the
same way Drive links become Drive stubs. A separate fetcher fills in the
`extracted_text` field later.

The denylist is conservative — we err on the side of skipping. Marketing
trackers, social profile pages, binary assets, and calendar invites are
all noise for a summarizer. Drive / Gmail URLs are handled by their
own pipelines so we don't double-stub them.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Strict-enough URL regex. Stops at whitespace, quotes, angle-brackets.
# Parens + square-brackets are allowed *inside* the URL (Wikipedia and
# others use them — e.g. `/wiki/Foo_(disambiguation)`). Trailing
# parens/brackets are then balanced off in `_strip_trailing_punct`.
# Doesn't try to be RFC-perfect — the denylist + ipaddress check in the
# fetcher are the real defense.
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+",
    re.IGNORECASE,
)

# Plain prose punctuation always stripped from the right side.
_TRAILING_PUNCT = ".,;:!?\"'>"


# Hosts whose sole purpose is email click tracking / list management.
# Fetching these gives us a redirect page or a 404 — never real content.
_DENY_HOSTS_EXACT = {
    "ct.sendgrid.net",
    "click.sendgrid.net",
    "links.sendgrid.net",
    "mail.google.com",
    "accounts.google.com",
    "goo.gl",
    "bit.ly",
    "t.co",
    "lnkd.in",
    "fb.me",
}

# Prefix matches on the host. Used for click-tracker subdomain
# patterns that don't anchor to a TLD — e.g. "click.mail.*" matches
# click.mail.vendor.com.
_DENY_HOST_PREFIXES = (
    "click.mail.",
    "link.email.",
    "click.e.",
    "trk.klclick",
)

# Domain suffix matches. An entry X matches a host Y if Y == X or Y
# ends with ".X" — i.e. proper domain / subdomain match, NOT a bare
# substring. This avoids flagging `notlinkedin.example.com` on a
# `linkedin.com` rule.
_DENY_DOMAIN_SUFFIXES = (
    "list-manage.com",
    "hubspotlinks.com",
    "constantcontact.com",
    "email.salesforce.com",
    # Drive is handled by gmail/drive.py — don't double-stub.
    "docs.google.com",
    "drive.google.com",
    # Social profile pages are almost never useful context for a
    # summarizer, and they tend to require login to fetch anything
    # meaningful anyway. Err on the side of skipping.
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
)

# URL-path / query-string substrings that indicate unsubscribe /
# preferences pages. These are almost always per-recipient tokens that
# would 404 or actually unsubscribe us if we crawled them.
_DENY_PATH_CONTAINS = (
    "unsubscribe",
    "optout",
    "opt-out",
    "opt_out",
    "manage-subscription",
    "manage_preferences",
    "email-preferences",
    "email_preferences",
    "preferences_center",
)

# Binary / non-textual extensions. We let the main content-type check
# in the fetcher double-guard, but skipping on extension means we don't
# even DNS-lookup an obviously-binary URL.
_DENY_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".ico",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mp3",
    ".wav",
    ".ogg",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".ics",
    ".zip",
    ".tar",
    ".gz",
    ".dmg",
    ".exe",
    ".apk",
)


def _strip_trailing_punct(url: str) -> str:
    """Trim trailing prose punctuation that isn't part of the URL.

    Parens and square brackets are handled via balance, not a flat
    strip — we want to keep `Foo_(disambiguation)` but lose the `)` in
    `(see https://example.com)`. Rule: drop a trailing close-bracket
    only when it has no matching opener anywhere earlier in the URL.
    """
    changed = True
    while url and changed:
        changed = False
        if url[-1] in _TRAILING_PUNCT:
            url = url[:-1]
            changed = True
            continue
        # Balanced trims for ) ] }.
        for close_ch, open_ch in ((")", "("), ("]", "["), ("}", "{")):
            if url.endswith(close_ch) and url.count(open_ch) < url.count(close_ch):
                url = url[:-1]
                changed = True
                break
    return url


def _host_is_denied(host: str) -> bool:
    host = host.lower()
    if host in _DENY_HOSTS_EXACT:
        return True
    if any(host.startswith(p) for p in _DENY_HOST_PREFIXES):
        return True
    return any(host == s or host.endswith("." + s) for s in _DENY_DOMAIN_SUFFIXES)


def _path_is_denied(path_and_query: str) -> bool:
    lower = path_and_query.lower()
    return any(token in lower for token in _DENY_PATH_CONTAINS)


def _suffix_is_denied(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in _DENY_SUFFIXES)


def _is_denied(url: str) -> bool:
    """True if the URL should be skipped entirely.

    Order: cheapest checks first (host match, then path/suffix) so we
    bail before doing any URL parsing beyond what urlparse already did.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return True
    host = parsed.hostname or ""
    if not host:
        return True
    if _host_is_denied(host):
        return True
    path_and_query = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    if _path_is_denied(path_and_query):
        return True
    if _suffix_is_denied(parsed.path):
        return True
    return False


def extract_crawlable_urls(body_text: str) -> list[str]:
    """Return a de-duplicated, order-preserving list of crawlable URLs
    extracted from `body_text`. URLs that hit the denylist are dropped.

    Matches the shape of `extract_drive_ids` — pure function, callers
    do the side-effecting upsert.
    """
    if not body_text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _URL_RE.finditer(body_text):
        candidate = _strip_trailing_punct(m.group(0))
        if not candidate:
            continue
        if candidate in seen:
            continue
        if _is_denied(candidate):
            seen.add(candidate)
            continue
        seen.add(candidate)
        out.append(candidate)
    return out
