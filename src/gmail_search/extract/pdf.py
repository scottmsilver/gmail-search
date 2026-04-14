import logging
from pathlib import Path
from typing import Any

import fitz  # pymupdf

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)


def extract_pdf(file_path: Path, config: dict[str, Any]) -> ExtractResult:
    max_pages = config.get("max_pdf_pages", 20)

    doc = fitz.open(str(file_path))

    text_parts: list[str] = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            text_parts.append(text.strip())

    full_text = "\n\n".join(text_parts) if text_parts else None

    images: list[Path] = []
    images_dir = file_path.parent / f"{file_path.stem}_pages"
    images_dir.mkdir(exist_ok=True)

    for i, page in enumerate(doc):
        if i >= max_pages:
            logger.info(f"Reached page limit ({max_pages}), skipping remaining pages for image rendering")
            break
        mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
        pix = page.get_pixmap(matrix=mat)
        img_path = images_dir / f"page_{i + 1:04d}.png"
        pix.save(str(img_path))
        images.append(img_path)

    doc.close()
    return ExtractResult(text=full_text, images=images)
