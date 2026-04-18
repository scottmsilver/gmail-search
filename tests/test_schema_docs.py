"""TABLE_DOCS must mirror SCHEMA. This catches the contact_frequency-style
drift that previously had the LLM querying columns that didn't exist."""

from __future__ import annotations

import pytest

from gmail_search.store.db import (
    _INTERNAL_TABLES,
    SCHEMA,
    TABLE_DOCS,
    _schema_table_names,
    assert_table_docs_cover_schema,
    describe_schema_for_llm,
)


def test_assertion_passes_today():
    assert_table_docs_cover_schema()


def test_every_documented_table_actually_exists_in_schema():
    schema_tables = _schema_table_names()
    for tbl in TABLE_DOCS:
        assert tbl in schema_tables, f"TABLE_DOCS has '{tbl}' but no CREATE TABLE found"


def test_every_user_facing_schema_table_is_documented():
    schema_tables = _schema_table_names() - _INTERNAL_TABLES
    for tbl in schema_tables:
        assert tbl in TABLE_DOCS, f"SCHEMA has '{tbl}' but TABLE_DOCS does not"


def test_internal_tables_actually_exist_in_schema_or_are_fts_shadow():
    # Sanity: every entry in _INTERNAL_TABLES is either in SCHEMA or is an
    # FTS5 shadow table created automatically by SQLite.
    schema_tables = _schema_table_names()
    fts_shadow_suffixes = ("_data", "_idx", "_docsize", "_config")
    for tbl in _INTERNAL_TABLES:
        if tbl in schema_tables:
            continue
        assert tbl.endswith(
            fts_shadow_suffixes
        ), f"_INTERNAL_TABLES has '{tbl}' which is neither in SCHEMA nor an FTS shadow"


def test_schema_section_headers_present():
    md = describe_schema_for_llm()
    assert md.startswith("### messages\n"), "messages should be the first table doc"
    for tbl in TABLE_DOCS:
        assert f"### {tbl}\n" in md, f"missing section for {tbl} in describe_schema_for_llm output"


def test_no_TODO_or_FIXME_in_docs():
    for tbl, doc in TABLE_DOCS.items():
        assert "TODO" not in doc.upper(), f"TODO leaked into TABLE_DOCS[{tbl}]"
        assert "FIXME" not in doc.upper(), f"FIXME leaked into TABLE_DOCS[{tbl}]"


def test_drift_detection_actually_fires(monkeypatch):
    # Inject a fake table that the docs don't cover; assertion must catch it.
    fake_schema = SCHEMA + "\nCREATE TABLE IF NOT EXISTS fake_drift (id TEXT);"
    monkeypatch.setattr("gmail_search.store.db.SCHEMA", fake_schema)
    with pytest.raises(RuntimeError, match="fake_drift"):
        assert_table_docs_cover_schema()


def test_stale_table_doc_is_caught(monkeypatch):
    # Add an entry to TABLE_DOCS that isn't in SCHEMA — assertion should flag.
    bogus = dict(TABLE_DOCS)
    bogus["nonexistent_table"] = "fake docs"
    monkeypatch.setattr("gmail_search.store.db.TABLE_DOCS", bogus)
    with pytest.raises(RuntimeError, match="nonexistent_table"):
        assert_table_docs_cover_schema()
