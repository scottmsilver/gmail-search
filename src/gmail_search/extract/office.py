"""Extract text from Office documents (docx, xlsx, csv)."""

import logging
from pathlib import Path
from typing import Any

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)


def extract_docx(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    """Extract paragraph text from Word documents."""
    from docx import Document

    try:
        doc = Document(str(file_path))
    except Exception as e:
        logger.warning(f"Failed to open docx {file_path}: {e}")
        return None

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return None

    return ExtractResult(text="\n\n".join(paragraphs))


def extract_xlsx(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    """Extract cell text from Excel spreadsheets."""
    from openpyxl import load_workbook

    try:
        wb = load_workbook(str(file_path), read_only=True, data_only=True)
    except Exception as e:
        logger.warning(f"Failed to open xlsx {file_path}: {e}")
        return None

    parts: list[str] = []
    max_rows = config.get("max_xlsx_rows", 500)

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        rows_text: list[str] = []
        for i, row in enumerate(sheet.iter_rows(values_only=True)):
            if i >= max_rows:
                break
            cells = [str(c) for c in row if c is not None]
            if cells:
                rows_text.append(" | ".join(cells))

        if rows_text:
            parts.append(f"Sheet: {sheet_name}\n" + "\n".join(rows_text))

    wb.close()

    if not parts:
        return None

    return ExtractResult(text="\n\n".join(parts))


def extract_csv(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    """Extract text from CSV files."""
    max_rows = config.get("max_xlsx_rows", 500)

    try:
        text = file_path.read_text(errors="replace")
        lines = text.strip().split("\n")[:max_rows]
        if not lines:
            return None
        return ExtractResult(text="\n".join(lines))
    except Exception as e:
        logger.warning(f"Failed to read CSV {file_path}: {e}")
        return None
