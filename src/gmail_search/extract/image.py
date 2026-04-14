from pathlib import Path
from typing import Any

from gmail_search.extract import ExtractResult


def extract_image(file_path: Path, config: dict[str, Any]) -> ExtractResult:
    return ExtractResult(text=None, images=[file_path])
