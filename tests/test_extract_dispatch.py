from pathlib import Path

from gmail_search.extract import dispatch


def test_dispatch_unknown_mime_returns_none():
    result = dispatch("application/zip", Path("/fake/path.zip"), {})
    assert result is None


def test_dispatch_image_passthrough(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0")
    result = dispatch("image/jpeg", img, {})
    assert result is not None
    assert result.text is None
    assert len(result.images) == 1
    assert result.images[0] == img


def test_dispatch_png_passthrough(tmp_path):
    img = tmp_path / "diagram.png"
    img.write_bytes(b"\x89PNG")
    result = dispatch("image/png", img, {})
    assert result is not None
    assert len(result.images) == 1
