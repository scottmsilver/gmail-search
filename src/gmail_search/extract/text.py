"""Extract text from plain text attachments."""

import logging
from pathlib import Path
from typing import Any

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)


def extract_text(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    """Read plain text files directly."""
    try:
        text = file_path.read_text(errors="replace").strip()
        if not text:
            return None
        return ExtractResult(text=text[:50000])
    except Exception as e:
        logger.warning(f"Failed to read text file {file_path}: {e}")
        return None


def extract_html(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    """Strip tags from an HTML file and keep readable text. Skips
    <script>/<style>/<noscript> so JS and CSS don't pollute the
    retrieval index.
    """
    try:
        raw = file_path.read_text(errors="replace")
    except Exception as e:
        logger.warning(f"Failed to read html {file_path}: {e}")
        return None
    text = html_to_text(raw)
    if len(text) < 80:
        return None
    return ExtractResult(text=text[:50000])


def html_to_text(html: str) -> str:
    """String-to-string variant of the HTML → text reducer. Same
    `<script>/<style>/<noscript>` filter, whitespace-collapsed output.
    Used by the summarizer as a fallback for emails whose `body_text`
    is empty but `body_html` carries the real content — a LOT of
    marketing / transactional mail is shaped that way.
    """
    if not html:
        return ""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())
