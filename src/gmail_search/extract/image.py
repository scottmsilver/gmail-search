import logging
from pathlib import Path
from typing import Any

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)


def extract_image(file_path: Path, config: dict[str, Any]) -> ExtractResult:
    """Passthrough for standard image formats (jpg, png, gif, webp)."""
    return ExtractResult(text=None, images=[file_path])


def extract_heic(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    """Convert HEIC/HEIF (iPhone photos) to JPEG for embedding."""
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        from PIL import Image

        img = Image.open(file_path)
        jpeg_path = file_path.with_suffix(".jpg")
        img.convert("RGB").save(jpeg_path, "JPEG", quality=85)
        img.close()
        return ExtractResult(text=None, images=[jpeg_path])
    except Exception as e:
        logger.warning(f"Failed to convert HEIC {file_path}: {e}")
        return None
