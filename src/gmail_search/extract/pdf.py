import logging
from pathlib import Path
from typing import Any

import fitz  # pymupdf

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)


def extract_pdf(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    max_pages = config.get("max_pdf_pages", 20)

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        logger.warning(f"Failed to open PDF {file_path}: {e}")
        return None

    if doc.is_encrypted:
        logger.warning(f"Skipping encrypted PDF: {file_path}")
        doc.close()
        return None

    max_text_pages = max_pages * 5  # extract text from more pages than images
    text_parts: list[str] = []
    for i in range(min(len(doc), max_text_pages)):
        try:
            page = doc[i]
            text = page.get_text()
            if text.strip():
                text_parts.append(text.strip())
        except Exception as e:
            logger.warning(f"Failed to extract text from page {i} of {file_path}: {e}")

    full_text = "\n\n".join(text_parts) if text_parts else None

    images: list[Path] = []
    images_dir = file_path.parent / f"{file_path.stem}_pages"
    images_dir.mkdir(exist_ok=True)

    for i in range(min(len(doc), max_pages)):
        try:
            page = doc[i]
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img_path = images_dir / f"page_{i + 1:04d}.png"
            pix.save(str(img_path))
            images.append(img_path)
        except Exception as e:
            logger.warning(f"Failed to render page {i} of {file_path}: {e}")

    if len(doc) > max_pages:
        logger.info(f"Reached page limit ({max_pages}), skipped {len(doc) - max_pages} pages for image rendering")

    doc.close()
    return ExtractResult(text=full_text, images=images)
