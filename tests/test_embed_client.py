from gmail_search.embed.client import estimate_tokens, format_message_text, truncate_to_token_limit


def test_estimate_tokens():
    text = "Hello world this is a test"
    tokens = estimate_tokens(text)
    assert 4 <= tokens <= 10


def test_truncate_to_token_limit():
    short = "Hello world"
    assert truncate_to_token_limit(short, 8192) == short

    long_text = "word " * 10000
    truncated = truncate_to_token_limit(long_text, 100)
    tokens = estimate_tokens(truncated)
    assert tokens <= 110


def test_format_message_text():
    text = format_message_text(
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        date="2025-06-15",
        subject="Hello",
        body="Body text here",
    )
    assert "From: alice@example.com" in text
    assert "To: bob@example.com" in text
    assert "Subject: Hello" in text
    assert "Body text here" in text


# ─── Image preprocessing (issue #12) ────────────────────────────────────
#
# Eight on-disk files failed every embedding pass with a Gemini 400 and
# were resubmitted forever. embed_image must now validate locally and
# only send PNG/JPEG bytes that the model accepts.

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from gmail_search.embed import client as client_mod
from gmail_search.embed.client import (
    MAX_IMAGE_PIXELS,
    GeminiEmbedder,
    InvalidImage,
    prepare_image_for_embedding,
)


def _write_png(path: Path, size=(40, 30), mode="RGB") -> Path:
    Image.new(mode, size, color=128).save(path, format="PNG")
    return path


def _make_embedder() -> GeminiEmbedder:
    emb = GeminiEmbedder.__new__(GeminiEmbedder)
    emb.model = "test-model"
    emb.dimensions = 8
    emb.task_type_document = "RETRIEVAL_DOCUMENT"
    emb.task_type_query = "RETRIEVAL_QUERY"
    emb.client = MagicMock()
    emb.client.models.embed_content.return_value.embeddings = [MagicMock(values=[0.1] * 8)]
    return emb


def _sent_blob(emb: GeminiEmbedder):
    call = emb.client.models.embed_content.call_args
    part = call.kwargs["contents"].parts[0]
    return part.inline_data.mime_type, part.inline_data.data


def test_garbage_bytes_raise_invalid_image_without_api_call(tmp_path):
    bad = tmp_path / "Outlook-Title.png"
    bad.write_bytes(b"\x93\x1f\x00garbage" * 12)
    emb = _make_embedder()
    with pytest.raises(InvalidImage):
        emb.embed_image(bad)
    emb.client.models.embed_content.assert_not_called()


def test_ascii_text_named_gif_raises_invalid_image(tmp_path):
    bad = tmp_path / "26.gif"
    bad.write_text("malformed - '0' is 1 wide, not 4")
    with pytest.raises(InvalidImage):
        prepare_image_for_embedding(bad)


def test_truncated_gif_raises_invalid_image(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (251, 298), color=(10, 20, 30)).save(buf, format="GIF")
    data = buf.getvalue()
    truncated = tmp_path / "image001.gif"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(InvalidImage):
        prepare_image_for_embedding(truncated)


def test_valid_png_is_sent_as_is(tmp_path):
    p = _write_png(tmp_path / "ok.png")
    emb = _make_embedder()
    emb.embed_image(p)
    mime, data = _sent_blob(emb)
    assert mime == "image/png"
    assert data == p.read_bytes()


def test_valid_jpeg_is_sent_as_jpeg(tmp_path):
    p = tmp_path / "photo.jpg"
    Image.new("RGB", (40, 30), color=(200, 10, 10)).save(p, format="JPEG")
    data, mime = prepare_image_for_embedding(p)
    assert mime == "image/jpeg"
    assert data == p.read_bytes()


def test_mime_comes_from_content_not_extension(tmp_path):
    # A PNG saved under a .tif name used to be labelled image/png only by
    # luck of the default; a JPEG under .png was mislabelled outright.
    p = tmp_path / "actually-a-jpeg.png"
    Image.new("RGB", (40, 30)).save(p, format="JPEG")
    _, mime = prepare_image_for_embedding(p)
    assert mime == "image/jpeg"


def test_two_frame_tiff_is_reencoded_as_png(tmp_path):
    frames = [Image.new("L", (60, 40), color=c) for c in (50, 200)]
    p = tmp_path / "Measurement foldable.tif"
    frames[0].save(p, format="TIFF", save_all=True, append_images=frames[1:])
    data, mime = prepare_image_for_embedding(p)
    assert mime == "image/png"
    out = Image.open(io.BytesIO(data))
    assert out.format == "PNG"
    assert out.size == (60, 40)
    assert out.getpixel((0, 0)) == 50  # first frame, not second


def test_gif_is_reencoded_as_png(tmp_path):
    p = tmp_path / "anim.gif"
    Image.new("P", (20, 20)).save(p, format="GIF")
    data, mime = prepare_image_for_embedding(p)
    assert mime == "image/png"
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_oversized_image_is_downscaled_below_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "MAX_IMAGE_PIXELS", 10_000)
    p = _write_png(tmp_path / "page_0001.png", size=(300, 200))  # 60k px
    data, mime = prepare_image_for_embedding(p)
    assert mime == "image/png"
    out = Image.open(io.BytesIO(data))
    w, h = out.size
    assert w * h <= 10_000
    assert abs(w / h - 1.5) < 0.05  # aspect ratio preserved


def test_decompression_bomb_sized_image_is_still_processed(tmp_path, monkeypatch):
    # Pillow refuses anything over 2x Image.MAX_IMAGE_PIXELS at open().
    # Our own cap must take over instead of the open() blowing up.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
    monkeypatch.setattr(client_mod, "MAX_IMAGE_PIXELS", 5_000)
    p = _write_png(tmp_path / "bomb.png", size=(200, 100))  # 20k px > 2*1000
    data, mime = prepare_image_for_embedding(p)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10**8)  # only so the test can read the result
    out = Image.open(io.BytesIO(data))
    assert out.size[0] * out.size[1] <= 5_000


def test_image_over_decode_ceiling_is_rejected_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "MAX_DECODE_PIXELS", 1_000)
    p = _write_png(tmp_path / "toobig.png", size=(200, 100))
    with pytest.raises(InvalidImage):
        prepare_image_for_embedding(p)


def test_default_cap_is_sane():
    assert 4_000_000 <= MAX_IMAGE_PIXELS <= 50_000_000


def test_transparent_pixels_are_composited_onto_white_when_reencoding(tmp_path):
    im = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
    im.putpixel((0, 0), (0, 0, 0, 0))  # fully transparent corner
    p = tmp_path / "alpha.tif"
    im.save(p, format="TIFF")
    data, _ = prepare_image_for_embedding(p)
    out = Image.open(io.BytesIO(data))
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((5, 5)) == (255, 0, 0)


def test_16bit_grayscale_is_scaled_not_clipped(tmp_path):
    im = Image.new("I;16", (20, 20), 32768)
    im.putpixel((0, 0), 65535)
    im.putpixel((1, 1), 0)
    p = tmp_path / "scan16.tif"
    im.save(p, format="TIFF")
    data, _ = prepare_image_for_embedding(p)
    out = Image.open(io.BytesIO(data))
    assert out.mode == "L"
    assert out.getpixel((0, 0)) == 255
    assert out.getpixel((1, 1)) == 0
    assert 120 <= out.getpixel((5, 5)) <= 136
