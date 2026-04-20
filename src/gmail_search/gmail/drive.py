"""Google Drive reader — fetches Docs/Sheets/Slides that are linked
from an email body as plain text via the Drive API. Uses the same
OAuth credentials as the Gmail client; if the stored token doesn't
carry drive.readonly scope, `build_drive_service` raises and the
caller should gracefully disable Drive enrichment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from googleapiclient.discovery import Resource, build

from gmail_search.gmail.auth import get_credentials

logger = logging.getLogger(__name__)

# URL-in-prose scanner. We need a broad find-URL-in-text regex
# because messages are free-form — `gdown.parse_url` works on a
# single URL, not a chunk of prose. Once we've carved out candidate
# URLs, delegate the (file_id, is_download) parsing to gdown —
# they maintain the canonical set of patterns (user-scoped /u/N/,
# /file/d/, /uc?id=, shared /presentation/d/, etc.) and the test
# matrix for Drive's URL shapes.
_URL_IN_TEXT = re.compile(
    r"https?://(?:docs|drive)\.google\.com/[^\s<>\"'\]]+",
    re.IGNORECASE,
)

_KIND_BY_PATH = {"document": "doc", "spreadsheets": "sheet", "presentation": "slides", "file": "file"}
_EXPORT_MIME = {
    # Google-native docs support export to text/plain directly.
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


def _kind_from_url(url: str) -> str:
    """Classify a parsed URL as doc/sheet/slides/file by inspecting
    the path segment. This is ours (not gdown's) because gdown's
    parse_url returns only the file_id, not the kind — but the
    classification is pure string work.
    """
    lower = url.lower()
    for kw, kind in _KIND_BY_PATH.items():
        if f"/{kw}/" in lower:
            return kind
    return "file"


def extract_drive_ids(body_text: str) -> list[tuple[str, str]]:
    """Return `[(drive_id, kind), ...]` unique by drive_id, preserving
    first-seen order. `kind` is one of 'doc', 'sheet', 'slides', 'file'.

    Parsing-strategy: find every Drive URL in the prose (broad regex),
    then hand each candidate to `gdown.parse_url.parse_url` which is
    the maintained library function for Drive-URL → file_id mapping
    (handles /u/N/, /uc?id=, /e/ publish links, /file/d/, etc.).
    """
    from gdown.parse_url import parse_url as _gdown_parse

    seen: dict[str, str] = {}
    for m in _URL_IN_TEXT.finditer(body_text or ""):
        url = m.group(0)
        drive_id, _is_download = _gdown_parse(url)
        if not drive_id or drive_id in seen:
            continue
        seen[drive_id] = _kind_from_url(url)
    return list(seen.items())


DRIVE_MIME_BY_KIND = {
    "doc": "application/vnd.google-apps.document",
    "sheet": "application/vnd.google-apps.spreadsheet",
    "slides": "application/vnd.google-apps.presentation",
    "file": "application/vnd.google-apps.file",
}


def drive_mime_for_kind(kind: str) -> str:
    """Pure mapping from our URL-category kind to Drive's mime type.
    Kept in the Drive client module because it's Drive-specific
    vocabulary, but has no side effects — safe for the store layer
    to import when building stub rows.
    """
    return DRIVE_MIME_BY_KIND.get(kind, "application/vnd.google-apps.file")


# Drive file IDs are 25-44 chars of `[A-Za-z0-9_-]`. Anything outside
# that charset in a stub filename means the row was tampered with —
# reject rather than pass through to the Drive API, which would
# otherwise happily fetch an attacker-chosen ID on the user's behalf.
# Codex audit 2026-04-20 flagged the original no-validation path.
_DRIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")


def _is_valid_drive_id(drive_id: str | None) -> bool:
    return bool(drive_id) and _DRIVE_ID_RE.match(drive_id) is not None


def drive_id_from_stub_filename(filename: str) -> str | None:
    """Extract the drive_id out of a `Drive: [<id>]` or
    `Drive: <title> [<id>]` filename. Returns None if the filename
    isn't a stub OR if the embedded ID doesn't match Drive's ID
    charset — a tampered row must not become an authority to fetch
    arbitrary Drive files.
    """
    if not filename or not filename.startswith("Drive:"):
        return None
    if not (filename.endswith("]") and "[" in filename):
        return None
    candidate = filename.rsplit("[", 1)[1].rstrip("]").strip()
    return candidate if _is_valid_drive_id(candidate) else None


_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


def build_drive_service(data_dir: Path) -> Resource:
    """Build a Drive v3 service using the same token.json as Gmail.

    Verifies the credentials carry `drive.readonly` and raises a
    `PermissionError` up-front if not. Without this check, missing
    scope only surfaces as an opaque 403 on the first Drive call,
    which hides the real cause from the logs. Codex audit 2026-04-20
    flagged the silent-fallback path.

    Callers should wrap this in try/except and skip Drive enrichment
    gracefully so a missing scope doesn't block the rest of the
    pipeline.
    """
    creds = get_credentials(data_dir)
    granted = set(getattr(creds, "scopes", None) or [])
    if _DRIVE_SCOPE not in granted:
        raise PermissionError(
            f"Stored credentials are missing scope {_DRIVE_SCOPE!r}. "
            "Delete data/token.json and re-auth to enable Drive features."
        )
    return build("drive", "v3", credentials=creds)


def fetch_doc_text(service: Resource, drive_id: str, *, max_chars: int = 50_000) -> tuple[str, str] | None:
    """Fetch one Drive item as plain text. Returns `(title, text)` or
    `None` on failure. Handles:

    - Native Google Docs → export as text/plain
    - Native Sheets      → export as CSV
    - Native Slides      → export as text/plain
    - Uploaded files      → get_media() and try to decode as UTF-8

    Quietly skips binary uploads (octet-stream) where decode would be
    meaningless — the caller's existing attachment flow handles those
    via download + mime-specific extractors.
    """
    try:
        meta = service.files().get(fileId=drive_id, fields="id, name, mimeType", supportsAllDrives=True).execute()
    except Exception as e:
        logger.warning(f"drive.get failed for {drive_id}: {e}")
        return None

    name = meta.get("name") or f"drive:{drive_id}"
    mime = meta.get("mimeType") or ""

    try:
        if mime in _EXPORT_MIME:
            blob = service.files().export_media(fileId=drive_id, mimeType=_EXPORT_MIME[mime]).execute()
        elif mime.startswith("text/") or mime == "application/json":
            blob = service.files().get_media(fileId=drive_id).execute()
        else:
            # Non-text Drive upload — skip, the attachment pipeline
            # handles these via the direct Gmail download instead.
            return None
    except Exception as e:
        logger.warning(f"drive.export failed for {drive_id} ({mime}): {e}")
        return None

    if isinstance(blob, bytes):
        text = blob.decode("utf-8", errors="replace")
    else:
        text = str(blob)
    text = text.strip()
    if not text:
        return None
    return name, text[:max_chars]
