from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractResult:
    text: str | None = None
    images: list[Path] = field(default_factory=list)


# Extension → mime redispatch for misreported `application/octet-stream`.
# We observed 145 PDFs, 30 ICS, and 50 PNGs hiding behind this generic
# mime in the corpus — a sizable chunk of missing extraction.
_EXT_TO_MIME = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "ics": "text/calendar",
    "csv": "text/csv",
    "txt": "text/plain",
    "html": "text/html",
    "htm": "text/html",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "eml": "message/rfc822",
}


def _sniff_mime_from_filename(file_path: Path) -> str | None:
    ext = file_path.suffix.lstrip(".").lower()
    return _EXT_TO_MIME.get(ext)


def dispatch(mime_type: str, file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    from gmail_search.extract.archive import extract_zip
    from gmail_search.extract.calendar import extract_calendar
    from gmail_search.extract.image import extract_heic, extract_image
    from gmail_search.extract.office import extract_csv, extract_doc, extract_docx, extract_pptx, extract_xlsx
    from gmail_search.extract.pdf import extract_pdf
    from gmail_search.extract.text import extract_html, extract_text

    extractors = {
        # PDF
        "application/pdf": extract_pdf,
        # Images (returned as image_path for Gemini multimodal embedding)
        "image/jpeg": extract_image,
        "image/jpg": extract_image,
        "image/png": extract_image,
        "image/gif": extract_image,
        "image/webp": extract_image,
        "image/tiff": extract_image,
        "image/heic": extract_heic,
        "image/heif": extract_heic,
        # Office documents
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": extract_docx,
        "application/msword": extract_doc,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": extract_xlsx,
        "application/vnd.ms-excel": extract_xlsx,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": extract_pptx,
        "text/csv": extract_csv,
        # Calendar
        "text/calendar": extract_calendar,
        "application/ics": extract_calendar,
        # Text / markup
        "text/plain": extract_text,
        "text/html": extract_html,
        # Archives
        "application/zip": extract_zip,
        "application/x-zip-compressed": extract_zip,
    }

    # If the reported mime is the generic octet-stream, sniff by
    # filename extension and retry against the mapped mime.
    if mime_type == "application/octet-stream":
        sniffed = _sniff_mime_from_filename(file_path)
        if sniffed is not None:
            mime_type = sniffed

    extractor = extractors.get(mime_type)
    if extractor is None:
        return None

    return extractor(file_path, config)
