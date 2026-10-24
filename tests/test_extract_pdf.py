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


def test_render_caps_longest_side(tmp_path):
    # A poster-sized page (3000 x 2000 pt) at the old fixed 150 DPI became
    # 6250 x 4167 px; a real 52 x 35 inch page came out at 27,115 x 18,154
    # (492 Mpx) and Gemini rejected it forever (issue #12).
    from PIL import Image

    from gmail_search.extract.pdf import MAX_RENDER_SIDE_PX

    pdf_path = tmp_path / "poster.pdf"
    doc = fitz.open()
    page = doc.new_page(width=3000, height=2000)
    page.insert_text((72, 72), "poster")
    doc.save(str(pdf_path))
    doc.close()

    result = extract_pdf(pdf_path, {"max_pdf_pages": 20})
    w, h = Image.open(result.images[0]).size
    assert max(w, h) <= MAX_RENDER_SIDE_PX
    assert max(w, h) >= MAX_RENDER_SIDE_PX - 2  # scaled to the cap, not further
    assert abs(w / h - 1.5) < 0.01


def test_render_keeps_150_dpi_for_normal_pages(tmp_path):
    from PIL import Image

    pdf_path = tmp_path / "letter.pdf"
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text((72, 72), "letter")
    doc.save(str(pdf_path))
    doc.close()
    result = extract_pdf(pdf_path, {"max_pdf_pages": 20})
    w, h = Image.open(result.images[0]).size
    assert (w, h) == (1275, 1650)


def test_octet_stream_truncated_filename_sniffs_pdf_magic(tmp_path):
    from gmail_search.extract import dispatch

    # Simulates a scanner filename truncated at 80 chars with a dot mid-name and no .pdf suffix.
    pdf_path = tmp_path / "10152026_OWNER NAME(S) .Oa- Sc r-'l PROJECT ADDRESS 24299 Summerhill Ave, Los Al"
    _create_test_pdf(pdf_path, pages=1)
    result = dispatch("application/octet-stream", pdf_path, {"max_pdf_pages": 20, "ocr": False})
    assert result is not None
    assert result.text is not None
    assert "Page 1 content here" in result.text


def test_extract_pdf_ocr_fallback_enforces_zdr(tmp_path, monkeypatch):
    # Create an image-only PDF (no embedded text layer).
    img_pdf = tmp_path / "scan.pdf"
    src = fitz.open()
    p = src.new_page(width=400, height=200)
    p.insert_text((36, 72), "Scanned Estimate $51,500")
    pix = p.get_pixmap(dpi=100)
    png_bytes = pix.tobytes("png")
    src.close()

    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_image(page.rect, stream=png_bytes)
    doc.save(str(img_pdf))
    doc.close()

    captured_bodies = []

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Scanned Estimate $51,500"}}]}

    def _fake_post(self, url, *, json=None, headers=None, timeout=None):
        captured_bodies.append(json)
        return _FakeResp()

    monkeypatch.setattr("httpx.Client.post", _fake_post)
    result = extract_pdf(img_pdf, {"max_pdf_pages": 20, "openrouter_key": "test-or-key"})
    assert result is not None
    assert result.text == "Scanned Estimate $51,500"
    assert len(captured_bodies) == 1
    assert captured_bodies[0]["model"] == "google/gemini-3-flash-preview"
    assert captured_bodies[0]["provider"] == {"zdr": True, "data_collection": "deny"}
