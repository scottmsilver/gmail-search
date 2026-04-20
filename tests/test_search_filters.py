"""Integration tests for the structured-filter candidate pre-restriction.

These tests exercise the path in ``SearchEngine.search_threads`` where
``_resolve_candidate_msg_ids`` narrows the BM25 + vector search corpus
before the usual ranking pipeline runs. The failure mode they guard
against: the top-K ScaNN/BM25 pool not containing the user's filtered
subset, yielding zero results even when matches exist.
"""

import struct
from datetime import datetime
from unittest.mock import MagicMock

from gmail_search.config import load_config
from gmail_search.index.builder import build_index
from gmail_search.search.engine import SearchEngine
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_message


def _make_vec(dims, value):
    return struct.pack(f"{dims}f", *([value] * dims))


def _insert_message(conn, *, msg_id, thread_id, from_addr, subject, body, date, dims, emb_value):
    upsert_message(
        conn,
        Message(
            id=msg_id,
            thread_id=thread_id,
            from_addr=from_addr,
            to_addr="me@test.com",
            subject=subject,
            body_text=body,
            body_html="",
            date=date,
            labels=["INBOX"],
            history_id=1,
            raw_json="{}",
        ),
    )
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id=msg_id,
            attachment_id=None,
            chunk_type="message",
            chunk_text=body,
            embedding=_make_vec(dims, emb_value),
            model="test-model",
        ),
    )


def _build_engine(tmp_path, db_path, dims):
    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.9] * dims

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"
    cfg["embedding"]["dimensions"] = dims
    # Disable reranker + off-topic filter so they don't interfere with the
    # filter-under-test semantics.
    cfg["search"] = {"rerank": False}

    return SearchEngine(db_path, index_dir, cfg, embedder=mock_embedder)


def test_from_filter_narrows_corpus_to_matching_sender(db_backend, tmp_path):
    """`from:david` must restrict the search corpus to David's messages
    BEFORE ranking, so the 3-David-messages-in-500-thread-haystack case
    still surfaces David's mail instead of losing him to top-K truncation.
    """
    dims = 16
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    # Seed: many Alice/Bob "board notes" messages + a handful of David's.
    # All have the same embedding value so the vector scores are identical
    # — the only thing differentiating them is the from_addr. If the
    # post-filter-only code path survives, David's mail can be crowded
    # out of the ranked result set.
    for i in range(40):
        _insert_message(
            conn,
            msg_id=f"alice{i}",
            thread_id=f"tA{i}",
            from_addr=f"alice{i}@x.com",
            subject="board notes",
            body="board notes content",
            date=datetime(2025, 1, 1 + (i % 28)),
            dims=dims,
            emb_value=0.9,
        )
    for i in range(40):
        _insert_message(
            conn,
            msg_id=f"bob{i}",
            thread_id=f"tB{i}",
            from_addr=f"bob{i}@x.com",
            subject="board notes",
            body="board notes content",
            date=datetime(2025, 2, 1 + (i % 28)),
            dims=dims,
            emb_value=0.9,
        )
    for i in range(3):
        _insert_message(
            conn,
            msg_id=f"david{i}",
            thread_id=f"tD{i}",
            from_addr=f"David Smith <david{i}@x.com>",
            subject="board notes",
            body="board notes content",
            date=datetime(2025, 3, 1 + i),
            dims=dims,
            emb_value=0.9,
        )
    conn.close()

    engine = _build_engine(tmp_path, db_path, dims)
    results = engine.search_threads("board notes from:david", top_k=20, filter_offtopic=False)

    assert results, "from:david should match David's 3 threads"
    # Every match in every surviving thread should come from someone matching
    # 'david' — that's the whole point of the refactor.
    for t in results:
        assert t.matches, f"thread {t.thread_id} has no matches"
        for m in t.matches:
            assert "david" in m.from_addr.lower(), f"thread {t.thread_id} has non-David match: {m.from_addr}"


def test_from_filter_with_zero_matches_returns_empty(db_backend, tmp_path):
    """A structured filter that resolves to an empty candidate set must
    short-circuit to []. Previously this returned pools-of-other-stuff
    post-filtered-to-empty; the new path exits early and cheaply.
    """
    dims = 16
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    # Seed some messages that don't match the filter.
    for i in range(10):
        _insert_message(
            conn,
            msg_id=f"m{i}",
            thread_id=f"t{i}",
            from_addr=f"real{i}@x.com",
            subject="hello",
            body="hello world",
            date=datetime(2025, 1, 1 + i),
            dims=dims,
            emb_value=0.9,
        )
    conn.close()

    engine = _build_engine(tmp_path, db_path, dims)
    results = engine.search_threads(
        "hello from:nonexistent@zzz.example",
        top_k=20,
        filter_offtopic=False,
    )
    assert results == [], "filter matching zero rows must short-circuit"


def test_no_filter_path_returns_results_unchanged(db_backend, tmp_path):
    """Queries with no structured filters take the full-corpus path. The
    refactor shouldn't have changed this behaviour — we want the same
    shape of result as before (>0 hits, thread_ids populated, etc.).
    """
    dims = 16
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    for i in range(20):
        _insert_message(
            conn,
            msg_id=f"m{i}",
            thread_id=f"t{i}",
            from_addr=f"user{i}@x.com",
            subject=f"Subject {i}",
            body=f"Body {i}",
            date=datetime(2025, 1, 1 + i),
            dims=dims,
            emb_value=0.9,
        )
    conn.close()

    engine = _build_engine(tmp_path, db_path, dims)
    results = engine.search_threads("subject body", top_k=10, filter_offtopic=False)
    assert results, "no-filter query should still return results"
    assert all(r.thread_id for r in results)


def test_date_filter_restricts_to_window(db_backend, tmp_path):
    """Date operators (after:/before:) or explicit date_from/date_to must
    only surface messages in that window. Both the endpoint-level args and
    the parsed operators should flow through the same candidate resolver.
    """
    dims = 16
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    # Messages spread across 3 years. Only 2024 should make it through
    # `after:2024-01-01 before:2024-12-31`.
    spread = [
        ("m2023", datetime(2023, 6, 1)),
        ("m2024a", datetime(2024, 3, 15)),
        ("m2024b", datetime(2024, 9, 1)),
        ("m2025", datetime(2025, 2, 1)),
    ]
    for idx, (mid, d) in enumerate(spread):
        _insert_message(
            conn,
            msg_id=mid,
            thread_id=f"t_{mid}",
            from_addr=f"u{idx}@x.com",
            subject="quarterly report",
            body="quarterly report content",
            date=d,
            dims=dims,
            emb_value=0.9,
        )
    conn.close()

    engine = _build_engine(tmp_path, db_path, dims)
    results = engine.search_threads(
        "quarterly report after:2024-01-01 before:2024-12-31",
        top_k=20,
        filter_offtopic=False,
    )
    assert results, "should find 2024 messages"
    for t in results:
        for m in t.matches:
            # Date is stored as ISO; lexicographic compare works.
            assert "2024-01-01" <= m.date <= "2024-12-31T23:59:59", f"match {m.message_id} outside window: {m.date}"
