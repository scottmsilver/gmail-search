"""Phase 2 + 3a backfill + tighten: assign all existing per-user table
rows to the bootstrap user, then promote nullable user_id columns to
NOT NULL on the hot paths so the invariant is enforced going forward.

Run after pg_schema.sql has added the columns (which is done as part
of init_db). Idempotent — re-running on an already-backfilled DB is
a no-op (UPDATE … WHERE user_id IS NULL touches zero rows).

Usage:
    GMS_BOOTSTRAP_EMAIL=scottmsilver@gmail.com \\
        uv run python scripts/migrate_phase23.py

Bootstrap email must already exist in `users`. We fetch its user_id
and stamp every NULL user_id with that value.

Per the 0d benchmark, chunked backfill of 565k rows ran in ~14s.
Total wall time for all tables here is ~1 minute.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from gmail_search.store.db import get_connection

# Tables that hold per-user data + the column to backfill (always
# `user_id`). Order matters for FK validation later: parents first.
_TABLES_TO_BACKFILL: list[str] = [
    "conversations",
    "messages",
    "attachments",
    "embeddings",
    "costs",
    "thread_summary",
    "topics",
    "message_topics",
    "contact_frequency",
    "term_aliases",
    "message_summaries",
    "summary_failures",
    "model_battles",
    "agent_sessions",
    # scann_index_pointer is handled separately by _promote_scann_pointer
    # because its `id` column gets dropped during promotion — re-running
    # the chunk-backfill loop after that would fail looking up `id`.
]

# Subset where we want NOT NULL after backfill so a missed write site
# can't silently insert a NULL user_id row. Skipping NOT NULL on
# topics/term_aliases/contact_frequency until their PK promotion
# (which implicitly sets NOT NULL) lands below.
_TABLES_TO_NOT_NULL: list[str] = [
    "messages",
    "embeddings",
    "attachments",
    "costs",
    "thread_summary",
    "message_topics",
    "message_summaries",
    "summary_failures",
    "model_battles",
    "conversations",
    "agent_sessions",
]


def _bootstrap_user_id(conn) -> str:
    email = os.environ.get("GMS_BOOTSTRAP_EMAIL", "scottmsilver@gmail.com")
    row = conn.execute("SELECT id FROM users WHERE email = %s", (email,)).fetchone()
    if not row:
        raise SystemExit(
            f"no users row for {email!r} — invite + sign in first, or "
            "set GMS_BOOTSTRAP_EMAIL to a different existing user."
        )
    return row["id"]


def _chunked_backfill(conn, table: str, user_id: str, chunk: int = 5000) -> int:
    """Per the 0d benchmark, 5000-row chunks UPDATE in ~50ms each.
    Smaller chunks = more round trips, larger = longer write locks per
    iteration. 5000 was the sweet spot in the bench."""
    # Tables with composite or non-`id` PKs need a different chunking
    # column. The "id IN (SELECT id ... LIMIT N)" pattern only works
    # when there's a single-column unique key called `id`.
    chunk_keys = {
        "conversations": "id",
        "messages": "id",
        "attachments": "id",
        "embeddings": "id",
        "costs": "id",
        "model_battles": "id",
        "agent_sessions": "id",
        "message_summaries": "message_id",
        "summary_failures": "message_id",
        "thread_summary": "thread_id",
        "topics": "topic_id",
        "term_aliases": "term",
        "contact_frequency": "email",
        "message_topics": "ctid",  # composite PK; ctid is PG's row identity
    }
    key = chunk_keys.get(table, "id")
    total = 0
    while True:
        n = conn.execute(
            f"UPDATE {table} SET user_id = %s "
            f"WHERE {key} IN (SELECT {key} FROM {table} WHERE user_id IS NULL LIMIT {chunk})",
            (user_id,),
        ).rowcount
        conn.commit()
        if n == 0:
            break
        total += n
    return total


def _set_not_null(conn, table: str) -> None:
    conn.execute(f"ALTER TABLE {table} ALTER COLUMN user_id SET NOT NULL")
    conn.commit()


def _promote_pk(conn, table: str, new_keys: tuple[str, ...]) -> None:
    """Rebuild the PK to include user_id. Idempotent: only flips when
    the existing PK is the legacy single-column shape.

    Drops dependent FKs first (e.g. message_topics.topic_id → topics)
    and recreates them on the new composite PK afterwards. Wraps the
    whole sequence in one transaction so a partial failure rolls back
    cleanly instead of leaving a dropped FK behind."""
    row = conn.execute(
        "SELECT array_length(c.conkey, 1) AS n "
        "FROM pg_constraint c JOIN pg_class t ON c.conrelid = t.oid "
        "WHERE t.relname = %s AND c.contype = 'p'",
        (table,),
    ).fetchone()
    if row is None or row["n"] is None or row["n"] != 1:
        # Already composite (or no PK at all — leave alone).
        return

    # Find every FK that references this table's PK so we can drop +
    # recreate them. This isn't general-purpose; it only handles the
    # specific dependency we know about (message_topics.topic_id →
    # topics.topic_id). Other tables here have no inbound FKs.
    dependent_fks: list[tuple[str, str, str]] = []  # (child_table, fk_name, child_col)
    if table == "topics":
        dependent_fks.append(("message_topics", "message_topics_topic_id_fkey", "topic_id"))

    for child_table, fk_name, _ in dependent_fks:
        conn.execute(f"ALTER TABLE {child_table} DROP CONSTRAINT IF EXISTS {fk_name}")
    conn.execute(f"ALTER TABLE {table} ALTER COLUMN user_id SET NOT NULL")
    conn.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_pkey")
    keys = ", ".join(new_keys)
    conn.execute(f"ALTER TABLE {table} ADD PRIMARY KEY ({keys})")
    # Recreate FKs against the new composite PK (child_table must
    # have a matching user_id column; the schema added it earlier).
    for child_table, fk_name, child_col in dependent_fks:
        conn.execute(
            f"ALTER TABLE {child_table} "
            f"ADD CONSTRAINT {fk_name} "
            f"FOREIGN KEY (user_id, {child_col}) REFERENCES {table} (user_id, {child_col})"
        )
    conn.commit()


def _promote_scann_pointer(conn) -> None:
    """scann_index_pointer was single-row (CHECK(id=1)). Make it one
    row per user with user_id as the PK. Idempotent guard via column
    existence — once we've dropped `id`, skip."""
    row = conn.execute(
        "SELECT 1 FROM information_schema.columns " "WHERE table_name = 'scann_index_pointer' AND column_name = 'id'"
    ).fetchone()
    if not row:
        return
    conn.execute("ALTER TABLE scann_index_pointer ALTER COLUMN user_id SET NOT NULL")
    conn.execute("ALTER TABLE scann_index_pointer DROP CONSTRAINT IF EXISTS scann_index_pointer_pkey")
    conn.execute("ALTER TABLE scann_index_pointer DROP COLUMN id")
    conn.execute("ALTER TABLE scann_index_pointer ADD PRIMARY KEY (user_id)")
    conn.commit()


def main() -> None:
    db_path = Path("data") / "gmail_search.db"
    conn = get_connection(db_path)
    try:
        user_id = _bootstrap_user_id(conn)
        print(f"backfilling all NULL user_id rows to {user_id}")

        for table in _TABLES_TO_BACKFILL:
            t0 = time.perf_counter()
            n = _chunked_backfill(conn, table, user_id)
            elapsed = time.perf_counter() - t0
            tag = "  →" if n > 0 else "  ✓"
            print(f"{tag} {table:24} {n:>8,} rows  ({elapsed:.2f}s)")

        print("\npromoting hot-path columns to NOT NULL")
        for table in _TABLES_TO_NOT_NULL:
            try:
                _set_not_null(conn, table)
                print(f"  ✓ {table}.user_id NOT NULL")
            except Exception as exc:
                # Already NOT NULL → idempotent re-run; ignore.
                print(f"  · {table}.user_id (skipped: {exc})")

        print("\npromoting composite PKs (user_id, …)")
        for table, new_keys in [
            ("contact_frequency", ("user_id", "email")),
            ("term_aliases", ("user_id", "term")),
            ("topics", ("user_id", "topic_id")),
        ]:
            try:
                _promote_pk(conn, table, new_keys)
                print(f"  ✓ {table}.PK = {new_keys}")
            except Exception as exc:
                print(f"  · {table} PK (skipped: {exc})")

        print("\npromoting scann_index_pointer to per-user shape")
        try:
            _promote_scann_pointer(conn)
            print("  ✓ scann_index_pointer.PK = (user_id)")
        except Exception as exc:
            print(f"  · scann_index_pointer (skipped: {exc})")

        print("\ndone")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
