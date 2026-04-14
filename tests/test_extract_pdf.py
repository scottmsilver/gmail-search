from pathlib import Path

import fitz  # pymupdf

from gmail_search.extract.pdf import extract_pdf


def _create_test_pdf(path: Path, pages: int = 3) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} content here")
    doc.save(str(path))
    doc.close()


def test_extract_pdf_text(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _create_test_pdf(pdf_path, pages=2)
    result = extract_pdf(pdf_path, {"max_pdf_pages": 20})
    assert result.text is not None
    assert "Page 1" in result.text
    assert "Page 2" in result.text


def test_extract_pdf_images(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _create_test_pdf(pdf_path, pages=3)
    result = extract_pdf(pdf_path, {"max_pdf_pages": 20})
    assert len(result.images) == 3
    for img_path in result.images:
        assert img_path.exists()
        assert img_path.suffix == ".png"


def test_extract_pdf_respects_page_limit(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _create_test_pdf(pdf_path, pages=10)
    result = extract_pdf(pdf_path, {"max_pdf_pages": 3})
    assert len(result.images) == 3
    assert "Page 10" in result.text
