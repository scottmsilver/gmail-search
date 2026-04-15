from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractResult:
    text: str | None = None
    images: list[Path] = field(default_factory=list)


def dispatch(mime_type: str, file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    from gmail_search.extract.archive import extract_zip
    from gmail_search.extract.calendar import extract_calendar
    from gmail_search.extract.image import extract_heic, extract_image
    from gmail_search.extract.office import extract_csv, extract_docx, extract_xlsx
    from gmail_search.extract.pdf import extract_pdf
    from gmail_search.extract.text import extract_text

    extractors = {
        # PDF
        "application/pdf": extract_pdf,
        # Images
        "image/jpeg": extract_image,
        "image/jpg": extract_image,
        "image/png": extract_image,
        "image/gif": extract_image,
        "image/webp": extract_image,
        "image/heic": extract_heic,
        "image/heif": extract_heic,
        # Office documents
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": extract_docx,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": extract_xlsx,
        "application/vnd.ms-excel": extract_xlsx,
        "text/csv": extract_csv,
        # Calendar
        "text/calendar": extract_calendar,
        "application/ics": extract_calendar,
        # Plain text
        "text/plain": extract_text,
        # Archives
        "application/zip": extract_zip,
        "application/x-zip-compressed": extract_zip,
    }

    extractor = extractors.get(mime_type)
    if extractor is None:
        return None

    return extractor(file_path, config)
