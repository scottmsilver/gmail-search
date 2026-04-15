"""Extract text from files inside zip archives."""

import logging
import zipfile
from pathlib import Path
from typing import Any

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)

# Max files to extract from a single zip
MAX_FILES = 20
# Max total extracted text
MAX_TEXT_CHARS = 100000


def extract_zip(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    """Extract text from files inside a zip archive.

    Opens the zip, extracts supported file types to a temp directory,
    and runs the appropriate extractor on each. Concatenates all text.
    """
    from gmail_search.extract import dispatch

    try:
        zf = zipfile.ZipFile(file_path, "r")
    except (zipfile.BadZipFile, Exception) as e:
        logger.warning(f"Failed to open zip {file_path}: {e}")
        return None

    extract_dir = file_path.parent / f"{file_path.stem}_zip"
    extract_dir.mkdir(exist_ok=True)

    text_parts: list[str] = []
    images: list[Path] = []
    files_processed = 0

    for info in zf.infolist():
        if info.is_dir():
            continue
        if files_processed >= MAX_FILES:
            logger.info(f"Reached file limit ({MAX_FILES}) in zip {file_path.name}")
            break

        # Skip very large files inside the zip
        if info.file_size > config.get("max_file_size_mb", 10) * 1024 * 1024:
            continue

        # Sanitize filename to prevent path traversal
        safe_name = Path(info.filename).name
        if not safe_name:
            continue

        # Extract the file
        try:
            extracted_path = extract_dir / safe_name
            with zf.open(info) as src, open(extracted_path, "wb") as dst:
                dst.write(src.read())
        except Exception as e:
            logger.warning(f"Failed to extract {info.filename} from zip: {e}")
            continue

        # Guess mime type from extension
        mime_type = _guess_mime_type(safe_name)
        if not mime_type:
            continue

        # Run the appropriate extractor
        result = None
        try:
            result = dispatch(mime_type, extracted_path, config)
        except Exception as e:
            logger.warning(f"Failed to extract {safe_name} from zip: {e}")

        if result:
            if result.text:
                text_parts.append(f"[{safe_name}]\n{result.text}")
            images.extend(result.images)
            files_processed += 1

    zf.close()

    if not text_parts and not images:
        return None

    full_text = "\n\n".join(text_parts)[:MAX_TEXT_CHARS] if text_parts else None
    return ExtractResult(text=full_text, images=images)


def _guess_mime_type(filename: str) -> str | None:
    """Map file extension to MIME type for supported formats."""
    ext = Path(filename).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".ics": "application/ics",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".heic": "image/heic",
        ".webp": "image/webp",
    }.get(ext)
