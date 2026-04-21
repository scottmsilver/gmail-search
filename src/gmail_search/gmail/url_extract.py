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
from pathlib import Path
from urllib.parse import urlparse

# Curated email-tracker host list sourced from disconnectme/disconnect-
# tracking-protection `services.json` (Email category only — the
# EmailAggressive category was too broad, e.g. flagged `adobe.com`
# whole). Loaded lazily and cached at module-import.
#
# Regenerate with `scripts/refresh_disconnect.py`. The list is ~213
# hosts and is bundled into the repo so the crawler works offline.
_DISCONNECT_HOSTS_FILE = Path(__file__).parent / "disconnect_email_hosts.txt"


def _load_disconnect_hosts() -> frozenset[str]:
    try:
        text = _DISCONNECT_HOSTS_FILE.read_text()
    except OSError:
        return frozenset()
    hosts = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hosts.add(line.lower())
    return frozenset(hosts)


_DISCONNECT_HOSTS = _load_disconnect_hosts()

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
# patterns that don't anchor to a TLD — e.g. "click.*" matches
# click.t.delta.com, click.adlmail.org, click.email.americastestkitchen.com.
#
# Aggressive on purpose: a full-corpus catch-up scan found that
# ~75% of the naive URL matches were click trackers / marketing
# dispatchers that redirect to the real destination. Crawling them
# wastes a browser render per hit and produces no content — the
# landing page is just "redirecting…" or a cookie-consent shell.
#
# If a real site happens to sit on a `click.*` / `tracker.*`
# subdomain it'll get caught too; that's an acceptable false-
# positive for this class of link.
_DENY_HOST_PREFIXES = (
    "ablink.mail.",
    "click.",
    "click-",
    "clicks.",
    "ct.",
    "e.",
    "em.",
    "enews.",
    "eventing.",
    "link.",
    "links.",
    "news.email.",
    "notifications.",
    "r.",
    "s2.",  # common image/pixel tracker subdomain (washingtonpost, etc.)
    "sg.",  # sendgrid + sendinblue notifications fall-through
    "t.",
    "tracker.",
    "tracking.",
    "trk.",
    "view.",
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
    # Bulk-email redirects / marketing dispatchers that the prefix
    # rules don't catch.
    "rs6.net",  # Constant Contact redirector (r20.rs6.net, etc.)
    "e2ma.net",  # Emma email service
    "adlmail.org",
    "mkt.com",
    "sendgrid.net",
    "mailgun.net",
    "sparkpostmail.com",
    "salesforce-experience.com",
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
    if host in _DISCONNECT_HOSTS:
        return True
    if any(host.startswith(p) for p in _DENY_HOST_PREFIXES):
        return True
    if any(host == s or host.endswith("." + s) for s in _DENY_DOMAIN_SUFFIXES):
        return True
    # Check domain-suffix match against the Disconnect list too (so
    # `open.mkt1248.com` gets caught even though only `mkt1248.com`
    # is in the list, when the list entry is a bare registrable).
    return any(host.endswith("." + d) for d in _DISCONNECT_HOSTS)


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


# Per-message URL caps.
#
# Bulk mail (Gmail categories PROMOTIONS / UPDATES / SOCIAL / FORUMS)
# with more than `_BULK_URL_CAP` URLs is almost always a newsletter or
# digest where the value was the curation itself — crawling every
# linked article burns hours and rarely surfaces anything useful for a
# summary. We drop the URL list entirely for those.
#
# Personal mail keeps a much looser cap; we only bail if the thing is
# pathological (link-bomb / obvious spam / auto-generated). Someone
# sharing a vacation recap with 30 links still deserves a crawl.
_BULK_URL_CAP = 10
_HARD_URL_CAP = 50

# Gmail categories that indicate bulk / automated mail.
_BULK_LABELS = frozenset(
    {
        "CATEGORY_PROMOTIONS",
        "CATEGORY_UPDATES",
        "CATEGORY_SOCIAL",
        "CATEGORY_FORUMS",
    }
)


def _looks_bulk(labels: object) -> bool:
    """True if the message's Gmail labels put it in a bulk category.

    Labels come from our DB as either a JSON-encoded list (string) or
    a decoded list — callers have historically passed both. Handle
    both without a separate json import in the hot path.
    """
    if not labels:
        return False
    if isinstance(labels, str):
        s = labels.strip()
        if not s:
            return False
        # Cheap: look for the category tokens as substrings. JSON
        # list encoding doesn't mangle these, and we don't want the
        # import cost of json.loads on every message.
        return any(cat in s for cat in _BULK_LABELS)
    if isinstance(labels, (list, tuple, set, frozenset)):
        return any(lab in _BULK_LABELS for lab in labels)
    return False


def extract_crawlable_urls(body_text: str, labels: object = None) -> list[str]:
    """Return a de-duplicated, order-preserving list of crawlable URLs
    extracted from `body_text`. URLs that hit the denylist are dropped.

    Two caps guard against digest / newsletter bombs:
      * Bulk mail (see `_BULK_LABELS`) with > `_BULK_URL_CAP` URLs
        returns an empty list — we don't stub any of them.
      * Anything over `_HARD_URL_CAP` returns empty regardless —
        even a personal email with 50+ URLs is almost certainly
        auto-generated.

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
    if len(out) > _HARD_URL_CAP:
        return []
    if _looks_bulk(labels) and len(out) > _BULK_URL_CAP:
        return []
    return out
