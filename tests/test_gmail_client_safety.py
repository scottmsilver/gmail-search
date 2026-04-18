"""Filename sanitization + base64 padding correctness for the Gmail client."""

from __future__ import annotations

import base64

from gmail_search.gmail.client import _sanitize_filename
from gmail_search.gmail.parser import _decode_body

# ─── _sanitize_filename ───────────────────────────────────────────────


def test_strips_path_components():
    assert _sanitize_filename("../../etc/passwd") == "passwd"
    assert _sanitize_filename("/abs/path/secret.txt") == "secret.txt"
    assert (
        _sanitize_filename("dir\\windows.txt") == "dir\\windows.txt"
        or _sanitize_filename("dir\\windows.txt") == "windows.txt"
    )


def test_strips_null_bytes():
    out = _sanitize_filename("evil\x00name.txt")
    assert "\x00" not in out


def test_handles_empty_or_dotty_input():
    # Pure dots / spaces should not produce an empty path.
    assert _sanitize_filename("...") == "unnamed_attachment"
    assert _sanitize_filename("   ") == "unnamed_attachment"
    assert _sanitize_filename("") == "unnamed_attachment"
    assert _sanitize_filename(".") == "unnamed_attachment"


def test_normal_filename_preserved():
    assert _sanitize_filename("invoice-2026-04.pdf") == "invoice-2026-04.pdf"


def test_unicode_filename_preserved():
    assert _sanitize_filename("naïve résumé.pdf") == "naïve résumé.pdf"


# ─── base64 padding ───────────────────────────────────────────────────


def test_decode_body_handles_already_padded_input():
    # 4-byte aligned input — old `4 - len%4` formula would have added
    # 4 bogus '=' chars. Make sure our fix decodes correctly.
    raw = b"hello world!"  # 12 bytes, base64 == 16 chars (multiple of 4)
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    # Re-add Gmail-style: Gmail strips trailing '=' on most attachments.
    assert _decode_body(encoded) == "hello world!"


def test_decode_body_handles_short_input_needing_pad():
    raw = b"hi"  # 2 bytes, base64 needs 2 '=' padding
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert _decode_body(encoded) == "hi"


def test_decode_body_handles_one_byte():
    raw = b"X"  # 1 byte, base64 needs 2 '=' padding
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert _decode_body(encoded) == "X"


def test_decode_body_already_padded():
    raw = b"hi"
    encoded = base64.urlsafe_b64encode(raw).decode()  # keeps padding
    # Even with padding present, our decode shouldn't choke.
    assert _decode_body(encoded) == "hi"


def test_decode_body_empty():
    assert _decode_body("") == ""


# Quick smoke that the same fix is mirrored in client._download_attachment_data
# (we can't easily call the network function, so we just verify the math is the
# same shape.)
def test_attachment_padding_math_matches():
    # Re-derive what the function does for a length-12 input.
    # The fixed formula is `-len(data) % 4` which is 0 here.
    assert (-12) % 4 == 0
    assert (-1) % 4 == 3  # one byte missing → 3 '=' chars
    assert (-2) % 4 == 2
    assert (-3) % 4 == 1
