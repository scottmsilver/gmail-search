from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractResult:
    text: str | None = None
    images: list[Path] = field(default_factory=list)


def dispatch(mime_type: str, file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    from gmail_search.extract.image import extract_image
    from gmail_search.extract.pdf import extract_pdf

    extractors = {
        "application/pdf": extract_pdf,
        "image/jpeg": extract_image,
        "image/jpg": extract_image,
        "image/png": extract_image,
        "image/gif": extract_image,
    }

    extractor = extractors.get(mime_type)
    if extractor is None:
        return None

    return extractor(file_path, config)
