import base64
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import time
from typing import Any

import fitz  # pymupdf
import httpx

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)

RENDER_DPI = 150
# Longest side of a rendered page image. A 52 x 35 inch poster page at a
# fixed 150 DPI became a 27,115 x 18,154 px (492 Mpx) PNG that the
# embedding API rejected on every pass (issue #12).
MAX_RENDER_SIDE_PX = 4000

OCR_MODEL = "google/gemini-3-flash-preview"
OCR_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OCR_PROMPT = (
    "Extract all text from this scanned document page verbatim, preserving headings, "
    "line items, amounts, dates, and any handwritten entries. Output only the extracted text."
)


def _ocr_page_image(client: httpx.Client, img_path: Path, api_key: str, model: str) -> str:
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    body = {
        "model": model,
        "provider": {"zdr": True, "data_collection": "deny"},
        "temperature": 0.0,
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = client.post(OCR_ENDPOINT, json=body, headers=headers, timeout=60.0)
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            choices = resp.json().get("choices") or []
            text = (choices[0].get("message") or {}).get("content") or "" if choices else ""
            return text.replace("\x00", "").strip()
        except Exception as e:
            if attempt == 2:
                logger.warning(f"OCR failed for {img_path}: {e}")
            else:
                time.sleep(2**attempt)
    return ""


def _ocr_pdf_images(images: list[Path], config: dict[str, Any]) -> str | None:
    if not images or not config.get("ocr", True):
        return None
    api_key = config.get("openrouter_key") or os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    model = config.get("ocr_model", OCR_MODEL)
    with httpx.Client(timeout=60.0) as client:
        with ThreadPoolExecutor(max_workers=min(4, len(images))) as pool:
            parts = list(pool.map(lambda p: _ocr_page_image(client, p, api_key, model), images))
    non_empty = [p for p in parts if p]
    return "\n\n".join(non_empty) if non_empty else None


def _render_matrix(page) -> "fitz.Matrix":
    """150 DPI, scaled down so the longest side fits MAX_RENDER_SIDE_PX."""
    scale = RENDER_DPI / 72
    longest_pt = max(page.rect.width, page.rect.height)
    if longest_pt * scale > MAX_RENDER_SIDE_PX:
        scale = MAX_RENDER_SIDE_PX / longest_pt
    return fitz.Matrix(scale, scale)


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
            pix = page.get_pixmap(matrix=_render_matrix(page))
            img_path = images_dir / f"page_{i + 1:04d}.png"
            pix.save(str(img_path))
            images.append(img_path)
        except Exception as e:
            logger.warning(f"Failed to render page {i} of {file_path}: {e}")

    if len(doc) > max_pages:
        logger.info(f"Reached page limit ({max_pages}), skipped {len(doc) - max_pages} pages for image rendering")

    doc.close()
    if full_text is None and images:
        full_text = _ocr_pdf_images(images, config)
    return ExtractResult(text=full_text, images=images)
