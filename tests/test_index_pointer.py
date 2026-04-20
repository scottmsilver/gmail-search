"""Tests for the DB-backed ScaNN-index pointer.

The pointer replaces filesystem-rename "atomic swap" as the way a
running reindex hands off to readers. Each build writes to a fresh
timestamped sibling of `index_dir` and updates a single row in
`scann_index_pointer`. Readers resolve through that row, so they
never see a torn, half-written build even during a reindex cycle.
"""

from __future__ import annotations

from pathlib import Path

from gmail_search.index.builder import build_index_sharded
from gmail_search.index.searcher import resolve_active_index_dir
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.queries import get_active_index_dir, set_active_index_dir


def _seed(db_path: Path, n: int = 80, dims: int = 16) -> None:
    """Populate enough embeddings for build_index_sharded to produce
    a real output rather than hitting the empty-index shortcut.
    """
    import struct
    from datetime import datetime

    from gmail_search.store.models import EmbeddingRecord, Message
    from gmail_search.store.queries import insert_embedding, upsert_message

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        for i in range(n):
            msg = Message(
                id=f"m{i:03d}",
                thread_id="t",
                from_addr="a@b",
                to_addr="c@d",
                subject="s",
                body_text="body",
                body_html="",
                date=datetime(2025, 1, 1),
                labels=[],
                history_id=1,
                raw_json="{}",
            )
            upsert_message(conn, msg)
            vec = [float(((i + k) % 7) - 3) / 4 for k in range(dims)]
            blob = struct.pack(f"<{dims}f", *vec)
            insert_embedding(
                conn,
                EmbeddingRecord(
                    id=None,
                    message_id=f"m{i:03d}",
                    attachment_id=None,
                    chunk_type="body",
                    chunk_text="body",
                    embedding=blob,
                    model="test-model",
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ─── pointer round-trip ───────────────────────────────────────────────────


def test_set_and_get_active_index_dir(db_backend):
    """Basic round-trip: set then read returns the same string.

    Runs against both SQLite and Postgres via the `db_backend` fixture —
    this is the cleanest proof that the pointer shim works uniformly
    across backends."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        assert get_active_index_dir(conn) is None
        set_active_index_dir(conn, "/tmp/scann__first")
        conn.commit()
        assert get_active_index_dir(conn) == "/tmp/scann__first"
        # Second set replaces the first row — one pointer, one truth.
        set_active_index_dir(conn, "/tmp/scann__second")
        conn.commit()
        assert get_active_index_dir(conn) == "/tmp/scann__second"
    finally:
        conn.close()


# ─── builder writes to a versioned dir + flips pointer ────────────────────


def test_build_index_sharded_writes_versioned_sibling(tmp_path):
    """The builder should NOT write to the exact path passed in — it
    writes a versioned sibling and updates the pointer. The returned
    path is the one that actually got written.
    """
    db_path = tmp_path / "db.sqlite"
    _seed(db_path, n=40, dims=16)
    index_dir = tmp_path / "scann_index"

    actual = build_index_sharded(db_path, index_dir, model="test-model", dimensions=16, shard_size=25)

    # Versioned sibling next to index_dir, prefixed by the name.
    assert actual != index_dir
    assert actual.parent == index_dir.parent
    assert actual.name.startswith(f"{index_dir.name}__")
    assert (actual / "manifest.json").exists()
    # Pointer row matches.
    conn = get_connection(db_path)
    try:
        assert get_active_index_dir(conn) == str(actual)
    finally:
        conn.close()


def test_reindex_gc_removes_old_versioned_dir(tmp_path):
    """Two consecutive builds: the old versioned dir gets removed once
    the new one is promoted, so the disk footprint doesn't grow
    without bound on a long-running watch loop.
    """
    db_path = tmp_path / "db.sqlite"
    _seed(db_path, n=40, dims=16)
    index_dir = tmp_path / "scann_index"

    first = build_index_sharded(db_path, index_dir, model="test-model", dimensions=16, shard_size=25)
    assert first.exists()

    second = build_index_sharded(db_path, index_dir, model="test-model", dimensions=16, shard_size=40)
    assert second.exists()
    assert second != first
    assert not first.exists(), "previous build should have been GC'd"


# ─── resolve_active_index_dir ─────────────────────────────────────────────


def test_resolve_uses_pointer_when_present(db_backend, tmp_path):
    """The resolver returns the pointer target when set, not the
    passed-in fallback — that's the whole point of the pointer.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    active = tmp_path / "scann_index__20260420_abc"
    active.mkdir()
    conn = get_connection(db_path)
    try:
        set_active_index_dir(conn, str(active))
        conn.commit()
    finally:
        conn.close()

    resolved = resolve_active_index_dir(db_path, tmp_path / "scann_index")
    assert resolved == active


def test_resolve_falls_back_when_pointer_missing(db_backend, tmp_path):
    """Fresh DB with no pointer row → resolver returns the fallback.
    Keeps legacy callers / tests working."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    fallback = tmp_path / "scann_index"
    resolved = resolve_active_index_dir(db_path, fallback)
    assert resolved == fallback


def test_resolve_falls_back_when_pointer_target_missing(tmp_path):
    """If the pointer references a path that no longer exists on disk
    (hand-edited, disk wipe, etc.), fall back to the default rather
    than returning a broken path that would 500 the searcher.
    """
    db_path = tmp_path / "db.sqlite"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        set_active_index_dir(conn, str(tmp_path / "definitely-not-there"))
        conn.commit()
    finally:
        conn.close()
    fallback = tmp_path / "scann_index"
    resolved = resolve_active_index_dir(db_path, fallback)
    assert resolved == fallback


def test_resolve_without_pointer_table(tmp_path):
    """A DB file that predates the pointer table (old migration) should
    still work — the resolver introspects for the table before querying.
    """
    import sqlite3

    db_path = tmp_path / "db.sqlite"
    # Empty sqlite file with no schema at all.
    conn = sqlite3.connect(db_path)
    conn.close()
    fallback = tmp_path / "scann_index"
    assert resolve_active_index_dir(db_path, fallback) == fallback


# ─── crash / partial-build protection ─────────────────────────────────────


def test_old_build_still_serves_during_reader_lookup(tmp_path):
    """The defining property: after the second build's pointer flip,
    a reader that resolves through the DB lands on the NEW dir. During
    the second build (before the pointer flips), it would have been
    the OLD dir. Either way, never a half-written build.

    We simulate the "during build" view by setting the pointer to
    the first-build path explicitly and verifying the resolver
    returns that even when a partially-written newer dir exists
    alongside.
    """
    db_path = tmp_path / "db.sqlite"
    _seed(db_path, n=40, dims=16)
    index_dir = tmp_path / "scann_index"

    first = build_index_sharded(db_path, index_dir, model="test-model", dimensions=16, shard_size=40)

    # Simulate a partially-written second build that hasn't updated
    # the pointer yet.
    phantom = index_dir.parent / f"{index_dir.name}__phantom_inflight"
    phantom.mkdir()
    (phantom / "shard_0").mkdir()  # but no manifest, no ids.json

    # Resolver still returns the first build's path.
    assert resolve_active_index_dir(db_path, index_dir) == first
