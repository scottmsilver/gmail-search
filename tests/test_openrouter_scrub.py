"""OpenRouter backend: surrogate scrubbing so un-encodable Gmail bodies
can't crash-loop the summarizer."""

from __future__ import annotations

import pytest

from gmail_search.llm.openrouter import _scrub_surrogates


def test_scrub_lone_surrogate_makes_encodable():
    # A lone high surrogate (U+D83D) is what broke message 17974fa2e139213d —
    # valid str, but str.encode('utf-8') raises UnicodeEncodeError on it.
    bad = "before\ud83dafter"
    with pytest.raises(UnicodeEncodeError):
        bad.encode("utf-8")
    cleaned = _scrub_surrogates(bad)
    # After scrubbing it round-trips through UTF-8 (what httpx's JSON does).
    cleaned.encode("utf-8")
    assert "before" in cleaned and "after" in cleaned


def test_scrub_leaves_clean_text_untouched():
    for s in ("normal text", "emoji 😀 ok", "accénts ✓", ""):
        assert _scrub_surrogates(s) == s
