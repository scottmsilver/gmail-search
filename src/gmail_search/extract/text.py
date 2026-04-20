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
    from bs4 import BeautifulSoup

    try:
        raw = file_path.read_text(errors="replace")
    except Exception as e:
        logger.warning(f"Failed to read html {file_path}: {e}")
        return None
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    if len(text) < 80:
        return None
    return ExtractResult(text=text[:50000])
