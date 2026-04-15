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
