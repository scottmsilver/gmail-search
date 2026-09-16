"""The SQL examples handed to the model must name the key the database has.

These strings are documentation, not executable paths, which is exactly why
they rotted: nothing failed when they said `search_id` against a database keyed
on `id`. What fails is downstream and much later — the model writes the example
back as SQL and gets `column "search_id" does not exist`.

Two checks, because one is not enough:

* a behavioural check that each renderer follows the selected profile, and
* a source check that no model-facing string hardcodes a key literal, which is
  what catches the next person who adds an example by copy-paste.
"""
from __future__ import annotations

import importlib
import pathlib
import re

import pytest

from gmail_search.store import schema_profile

# Every module that renders a BM25 example for the model.
_SOURCES = [
    "src/gmail_search/store/db.py",
    "src/gmail_search/server.py",
    "src/gmail_search/agents/analyst.py",
    "src/gmail_search/agents/mcp_tools_server.py",
]

_REPO = pathlib.Path(__file__).resolve().parents[1]


def _render_all(monkeypatch, profile_name: str) -> dict[str, str]:
    """Every model-facing BM25 example, rendered under one selected profile."""
    monkeypatch.setenv(schema_profile.SELECTION_ENV, profile_name)
    from gmail_search import server
    from gmail_search.agents import analyst, mcp_tools_server
    from gmail_search.store import db

    return {
        "table_docs": db.describe_schema_for_llm(),
        "like_rejection": server._bm25_required_error("subject"),
        "analyst": analyst.analyst_instruction(),
        "mcp_sql_tool": mcp_tools_server.sql_query_batch_desc(),
    }


@pytest.mark.parametrize(
    ("profile_name", "expected", "forbidden"),
    [
        ("text-key-v1", "id", "search_id"),
        ("text-partitioned-v1", "id", "search_id"),
        ("numeric-key-v1", "search_id", None),
    ],
)
def test_examples_render_the_selected_key(monkeypatch, profile_name, expected, forbidden):
    for name, text in _render_all(monkeypatch, profile_name).items():
        assert f"{expected} @@@" in text, f"{name} does not score on {expected!r}"
        if forbidden:
            assert forbidden not in text, f"{name} still names {forbidden!r}"


def test_attachment_examples_keep_their_own_key(monkeypatch):
    """Attachments are keyed on their own PK in every profile — only the
    message key moves. A profile change must not re-key attachments."""
    monkeypatch.setenv(schema_profile.SELECTION_ENV, "numeric-key-v1")
    from gmail_search import server

    assert "id @@@ 'filename:foo'" in server._bm25_required_error("filename")


def test_placeholder_is_fully_substituted(monkeypatch):
    """A missed placeholder is worse than a stale literal: the model is handed
    a column called `{bm25_key}`."""
    for text in _render_all(monkeypatch, "text-key-v1").values():
        assert "{bm25_key}" not in text


# `search_id` used as a BM25 key in an example. Only the *message* key moves
# between profiles — `attachments` and `propositions` genuinely key on `id` in
# every profile, so a bare `id @@@` is correct there and is not swept up.
_HARDCODED = re.compile(r"search_id\s+@@@|paradedb\.score\(search_id\)")


@pytest.mark.parametrize("relpath", _SOURCES)
def test_no_module_hardcodes_the_message_bm25_key(relpath):
    source = (_REPO / relpath).read_text()
    offenders = [
        f"{relpath}:{source[:m.start()].count(chr(10)) + 1}: {m.group(0)}"
        for m in _HARDCODED.finditer(source)
    ]
    assert not offenders, (
        "BM25 examples must render from schema_profile.selected_bm25_key(), not a "
        "literal — use the {bm25_key} placeholder or an f-string:\n  "
        + "\n  ".join(offenders)
    )


def test_the_messages_doc_is_a_template():
    """The one entry that must move with the profile. Asserting the placeholder
    is present is what makes the check above meaningful: without it, deleting
    every example would also pass."""
    from gmail_search.store.db import TABLE_DOCS

    assert TABLE_DOCS["messages"].count("{bm25_key}") >= 4


def test_selected_bm25_key_rejects_an_unknown_profile(monkeypatch):
    monkeypatch.setenv(schema_profile.SELECTION_ENV, "no-such-profile")
    with pytest.raises(schema_profile.SchemaProfileMismatch, match="no-such-profile"):
        schema_profile.selected_bm25_key()


@pytest.mark.parametrize(
    ("profile_name", "expected"), [("numeric-key-v1", "search_id"), ("text-key-v1", "id")]
)
def test_mcp_tool_description_follows_the_profile(monkeypatch, profile_name, expected):
    """Resolved per call, not at import — a process that configures the profile
    after this module loads must not get a description that disagrees with the
    rest of the app."""
    monkeypatch.setenv(schema_profile.SELECTION_ENV, profile_name)
    module = importlib.import_module("gmail_search.agents.mcp_tools_server")
    assert f"{expected} @@@" in module.sql_query_batch_desc()
