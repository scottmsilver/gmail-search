#!/usr/bin/env python3
"""One-shot bulk migration: SQLite ``data/gmail_search.db`` → Postgres.

Design
------
- **Per-table pipeline.** We iterate the explicit ``TABLES`` list in FK-safe
  order. For each table: optionally ``TRUNCATE``, stream rows out of SQLite in
  ``--batch-size`` chunks, and bulk-load them into Postgres using
  ``COPY {table} ({cols}) FROM STDIN`` via ``psycopg.Cursor.copy``. COPY is
  ~10x faster than executemany INSERTs and — crucially for the 5 GiB of
  embedding BYTEA payload — streams rows without buffering the whole table
  in the client.
- **Transactional per table.** Each table's copy runs inside a single
  psycopg transaction. Any exception rolls the whole table back so the
  target never ends up half-populated.
- **Generated columns skipped.** ``messages.fts`` and ``attachments.fts``
  are ``GENERATED ALWAYS AS ... STORED`` tsvectors in PG and must NOT
  appear in the COPY column list — Postgres computes them at INSERT time
  from subject/body/etc. We build the COPY column list from PG's
  ``information_schema.columns`` with ``is_generated = 'NEVER'`` so this
  is automatic.
- **Sequence rewind.** Tables with ``BIGSERIAL`` primary keys keep their
  SQLite ids verbatim (COPY into the serial column is fine). After the
  copy we call ``setval('{seq}', MAX(id))`` so future app inserts don't
  collide with the copied rows. Handled generically via
  ``pg_get_serial_sequence``.
- **Verification.** ``--verify`` runs row-count checks on both sides and
  samples 100 random rows per table, diffing every non-generated column.
  Count mismatch exits 1 with the first 10 IDs that disagree.
- **FK drift.** If a child row points at a parent row that isn't in the
  SQLite snapshot, we log a warning and skip the row rather than abort.

Known caveats
-------------
- Generated tsvector columns (``messages.fts``, ``attachments.fts``) are
  never written — Postgres recomputes them. If ranking differs between
  SQLite FTS5 bm25 and PG ``ts_rank_cd`` that's expected; see the
  "Behavior change we accept" note in ``pg_schema.sql``.
- Any SQLite row whose primary-key type is integer and whose value lands
  above 2^63-1 will overflow BIGINT. Not a concern for this corpus (all
  ids come from ``AUTOINCREMENT`` counters well under 2^40).
- Postgres sequences are rewound to ``MAX(id)``. If SQLite had holes and
  the app relies on monotonic ids, behavior is unchanged — ``nextval``
  just picks the next free slot.
- FK-order is enforced by the hardcoded ``TABLES`` list. Running with
  ``--tables`` to migrate a single child table assumes its parents are
  already present in PG.

CLI
---
    python scripts/migrate_sqlite_to_pg.py \
        --sqlite data/gmail_search.db \
        --dsn postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search \
        [--tables messages,embeddings,...] \
        [--batch-size 5000] \
        [--verify] \
        [--truncate-first]
"""

from __future__ import annotations

import argparse
import logging
import random
import sqlite3
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import psycopg

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

# Migration order. Children come after their parents so FKs resolve.
TABLES: list[str] = [
    "messages",
    "attachments",
    "embeddings",
    "message_summaries",
    "thread_summary",
    "topics",
    "message_topics",
    "contact_frequency",
    "term_aliases",
    "costs",
    "sync_state",
    "query_cache",
    "conversations",
    "conversation_messages",
    "model_battles",
    "job_progress",
    "scann_index_pointer",
]

# Tables with an auto-increment primary key whose sequence must be
# rewound after COPY so the next app INSERT doesn't collide with an
# already-used id.
BIGSERIAL_TABLES: dict[str, str] = {
    "attachments": "id",
    "embeddings": "id",
    "costs": "id",
    "conversation_messages": "id",
    "model_battles": "id",
}

# Primary-key column(s) per table — used by --verify to diff specific rows
# and to list "ids present in SQLite but missing in PG". For composite
# keys we concat the parts with '|' for display purposes only.
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "messages": ("id",),
    "attachments": ("id",),
    "embeddings": ("id",),
    "message_summaries": ("message_id",),
    "thread_summary": ("thread_id",),
    "topics": ("topic_id",),
    "message_topics": ("message_id", "topic_id"),
    "contact_frequency": ("email",),
    "term_aliases": ("term",),
    "costs": ("id",),
    "sync_state": ("key",),
    "query_cache": ("query_text", "model"),
    "conversations": ("id",),
    "conversation_messages": ("id",),
    "model_battles": ("id",),
    "job_progress": ("job_id",),
    "scann_index_pointer": ("id",),
}

log = logging.getLogger("migrate")


# ─────────────────────────────────────────────────────────────────────
# Progress bar (tqdm if available, else simple fallback)
# ─────────────────────────────────────────────────────────────────────


def _make_progress(total: int, desc: str):
    try:
        from tqdm import tqdm  # noqa: PLC0415

        return tqdm(total=total, desc=desc, unit="row", mininterval=0.5)
    except ImportError:
        return _SimpleProgress(total=total, desc=desc)


class _SimpleProgress:
    """Fallback when tqdm isn't installed. Prints a percent line every 5%."""

    def __init__(self, total: int, desc: str):
        self.total = max(total, 1)
        self.desc = desc
        self.done = 0
        self._last_pct = -1

    def update(self, n: int = 1) -> None:
        self.done += n
        pct = int(100 * self.done / self.total)
        if pct // 5 != self._last_pct // 5:
            self._last_pct = pct
            print(f"  [{self.desc}] {self.done}/{self.total} ({pct}%)", flush=True)

    def close(self) -> None:
        print(f"  [{self.desc}] done: {self.done}/{self.total}", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ─────────────────────────────────────────────────────────────────────
# Schema introspection (Postgres side)
# ─────────────────────────────────────────────────────────────────────


def _get_pg_copy_columns(pg_conn: psycopg.Connection, table: str) -> list[str]:
    """Return the list of PG columns we should write during COPY.

    Skips GENERATED columns (``is_generated = 'ALWAYS'``) because Postgres
    computes those at INSERT time — including them in a COPY column list
    raises ``cannot insert a non-DEFAULT value into column ...``.
    """
    sql = (
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() "
        "  AND table_name = %s "
        "  AND is_generated = 'NEVER' "
        "ORDER BY ordinal_position"
    )
    with pg_conn.cursor() as cur:
        cur.execute(sql, (table,))
        return [row[0] for row in cur.fetchall()]


def _get_pg_all_columns(pg_conn: psycopg.Connection, table: str) -> list[str]:
    """All PG columns, including generated ones — used by --verify to know
    which columns to ignore when diffing a sampled row."""
    sql = (
        "SELECT column_name, is_generated FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s "
        "ORDER BY ordinal_position"
    )
    with pg_conn.cursor() as cur:
        cur.execute(sql, (table,))
        return [(r[0], r[1]) for r in cur.fetchall()]  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────
# Schema introspection (SQLite side)
# ─────────────────────────────────────────────────────────────────────


def _get_sqlite_columns(sqlite_conn: sqlite3.Connection, table: str) -> list[str]:
    rows = sqlite_conn.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
    return [r[1] for r in rows]


# ─────────────────────────────────────────────────────────────────────
# FK parent lookup (for drift detection)
# ─────────────────────────────────────────────────────────────────────


def _get_pg_fk_parents(pg_conn: psycopg.Connection, table: str) -> list[tuple[str, str, str]]:
    """Return ``[(local_col, parent_table, parent_col), ...]`` for every
    outgoing FK on ``table``. Used only when ``--truncate-first`` is NOT
    set and drift is possible — we probe the parent table for existence
    before copying a child row.
    """
    sql = """
        SELECT kcu.column_name,
               ccu.table_name,
               ccu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
           AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
           AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = current_schema()
          AND tc.table_name = %s
    """
    with pg_conn.cursor() as cur:
        cur.execute(sql, (table,))
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]  # type: ignore[return-value]


def _load_existing_parent_ids(pg_conn: psycopg.Connection, parent_table: str, parent_col: str) -> set[Any]:
    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT "{parent_col}" FROM "{parent_table}"')
        return {r[0] for r in cur.fetchall()}


# ─────────────────────────────────────────────────────────────────────
# Row transform
# ─────────────────────────────────────────────────────────────────────


def _sqlite_row_to_pg_tuple(
    sqlite_row: sqlite3.Row,
    copy_cols: Sequence[str],
) -> tuple:
    """Project a sqlite Row onto ``copy_cols`` (PG column order).

    SQLite ``bytes`` map 1:1 to PG ``BYTEA`` via psycopg's default adapter,
    so no transform is needed for the ``embedding`` blobs. Integers,
    strings, floats, and NULL pass through. SQLite has no boolean type,
    but the PG schema happens to have no BOOLEAN columns either (all
    status flags are TEXT), so no 0/1 → True/False conversion is needed.
    """
    return tuple(sqlite_row[c] for c in copy_cols)


# ─────────────────────────────────────────────────────────────────────
# Per-table migration
# ─────────────────────────────────────────────────────────────────────


def _chunks(iterator: Iterator[sqlite3.Row], size: int) -> Iterable[list[sqlite3.Row]]:
    batch: list[sqlite3.Row] = []
    for row in iterator:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _count_sqlite(sqlite_conn: sqlite3.Connection, table: str) -> int:
    return sqlite_conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def _count_pg(pg_conn: psycopg.Connection, table: str) -> int:
    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        return cur.fetchone()[0]


def _truncate_table(pg_conn: psycopg.Connection, table: str) -> None:
    with pg_conn.cursor() as cur:
        cur.execute(f'TRUNCATE "{table}" RESTART IDENTITY CASCADE')


def _rewind_sequence(pg_conn: psycopg.Connection, table: str, pk_col: str) -> None:
    """Rewind ``{table}_{pk_col}_seq`` to ``MAX({pk_col})``.

    We resolve the sequence via ``pg_get_serial_sequence`` so we don't
    hardcode the ``{table}_{pk}_seq`` naming (PG isn't strict about it).
    ``setval(..., max, true)`` sets is_called = true so ``nextval`` returns
    max+1 on the next call. If the table is empty, ``setval`` with
    is_called = false and value 1 leaves it in fresh-sequence state.
    """
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT pg_get_serial_sequence(%s, %s)",
            (table, pk_col),
        )
        seq_row = cur.fetchone()
        if not seq_row or seq_row[0] is None:
            return  # no sequence (shouldn't happen for BIGSERIAL_TABLES)
        seq_name = seq_row[0]
        cur.execute(f'SELECT COALESCE(MAX("{pk_col}"), 0) FROM "{table}"')
        max_id = cur.fetchone()[0] or 0
        if max_id > 0:
            cur.execute("SELECT setval(%s, %s, true)", (seq_name, max_id))
        else:
            cur.execute("SELECT setval(%s, 1, false)", (seq_name,))
    log.info("  rewound sequence for %s.%s to %s", table, pk_col, max_id)


def _migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    table: str,
    batch_size: int,
    truncate_first: bool,
) -> int:
    """Copy one table from SQLite to PG inside a single PG transaction.
    Returns the number of rows copied.
    """
    copy_cols = _get_pg_copy_columns(pg_conn, table)
    if not copy_cols:
        log.warning("no columns found in PG for %s — skipping", table)
        return 0

    sqlite_cols = set(_get_sqlite_columns(sqlite_conn, table))
    # Any copy column that doesn't exist on the SQLite side gets NULL.
    # Primarily covers ``job_progress.start_completed`` on ancient DBs
    # that predate the ALTER TABLE backfill.
    missing_on_sqlite = [c for c in copy_cols if c not in sqlite_cols]
    if missing_on_sqlite:
        log.info("  %s columns absent in SQLite (will write NULL): %s", table, missing_on_sqlite)

    if truncate_first:
        _truncate_table(pg_conn, table)

    # FK drift detection: pre-load every parent-column id set so we can
    # filter orphans in O(1). Skipped if --truncate-first because the
    # user is explicitly saying "I want to start clean, don't worry
    # about drift" — and the sets would be empty anyway.
    parent_id_sets: dict[str, set[Any]] = {}
    fk_meta = _get_pg_fk_parents(pg_conn, table)
    if fk_meta and not truncate_first:
        for _, parent_table, parent_col in fk_meta:
            key = f"{parent_table}.{parent_col}"
            if key not in parent_id_sets:
                parent_id_sets[key] = _load_existing_parent_ids(pg_conn, parent_table, parent_col)

    total_rows = _count_sqlite(sqlite_conn, table)
    log.info("migrating %s: %d rows in SQLite", table, total_rows)
    if total_rows == 0:
        return 0

    # Stream from SQLite. We use a dedicated cursor so ``fetchmany``
    # iteration is explicit and the row factory stays as sqlite3.Row.
    select_cols = ", ".join(f'"{c}"' if c in sqlite_cols else "NULL" for c in copy_cols)
    # Re-alias NULL columns so sqlite3.Row indexing by column name works.
    # Strategy: fetch only the SQLite-present columns, then look up by
    # name in the transform step. If a column is missing on SQLite, the
    # transform fills NULL.
    present_cols = [c for c in copy_cols if c in sqlite_cols]
    present_sql = ", ".join(f'"{c}"' for c in present_cols)
    select_sql = f'SELECT {present_sql} FROM "{table}"'

    skipped_orphans = 0

    cols_sql = ", ".join(f'"{c}"' for c in copy_cols)
    copy_sql = f'COPY "{table}" ({cols_sql}) FROM STDIN'

    progress = _make_progress(total_rows, table)
    copied = 0
    try:
        with pg_conn.cursor() as cur:
            with cur.copy(copy_sql) as copy_stream:
                src_cur = sqlite_conn.cursor()
                src_cur.execute(select_sql)
                while True:
                    batch = src_cur.fetchmany(batch_size)
                    if not batch:
                        break
                    for row in batch:
                        if _orphan_row(row, fk_meta, parent_id_sets):
                            skipped_orphans += 1
                            progress.update(1)
                            continue
                        out_tuple = _project_row(row, present_cols, copy_cols)
                        copy_stream.write_row(out_tuple)
                        copied += 1
                        progress.update(1)
                src_cur.close()
        pg_conn.commit()
    except Exception:
        pg_conn.rollback()
        log.exception("failed to copy %s — rolled back", table)
        raise
    finally:
        progress.close()

    if skipped_orphans:
        log.warning("  %s: skipped %d orphan row(s) (FK parent missing)", table, skipped_orphans)
    log.info("  %s: copied %d rows", table, copied)

    # Rewind BIGSERIAL so the next INSERT doesn't collide with a copied id.
    if table in BIGSERIAL_TABLES:
        _rewind_sequence(pg_conn, table, BIGSERIAL_TABLES[table])
        pg_conn.commit()

    return copied


def _project_row(
    row: sqlite3.Row,
    present_cols: Sequence[str],
    copy_cols: Sequence[str],
) -> tuple:
    """Build the output tuple in PG's column order, filling NULL for any
    column that doesn't exist in SQLite."""
    present_set = set(present_cols)
    # row is a sqlite3.Row — indexable by column name.
    return tuple(row[c] if c in present_set else None for c in copy_cols)


def _orphan_row(
    row: sqlite3.Row,
    fk_meta: Sequence[tuple[str, str, str]],
    parent_id_sets: dict[str, set[Any]],
) -> bool:
    """True if this row's FK points at a parent that isn't in PG."""
    if not parent_id_sets:
        return False
    for local_col, parent_table, parent_col in fk_meta:
        # SQLite might not have this column in its schema (e.g. an FK we
        # added only on the PG side). Treat missing as "not an orphan".
        try:
            val = row[local_col]
        except (IndexError, KeyError):
            continue
        if val is None:
            continue
        key = f"{parent_table}.{parent_col}"
        if key not in parent_id_sets:
            continue
        if val not in parent_id_sets[key]:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────


def _verify_counts(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    table: str,
) -> bool:
    """True iff row counts match. On mismatch, log the first 10
    diverging primary-key values."""
    s_count = _count_sqlite(sqlite_conn, table)
    p_count = _count_pg(pg_conn, table)
    if s_count == p_count:
        log.info("  [verify] %s: count match (%d)", table, s_count)
        return True

    log.error("  [verify] %s: COUNT MISMATCH — sqlite=%d pg=%d", table, s_count, p_count)
    _log_id_diff(sqlite_conn, pg_conn, table)
    return False


def _log_id_diff(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    table: str,
) -> None:
    pk_cols = PRIMARY_KEYS.get(table)
    if not pk_cols:
        return
    pk_expr_sqlite = " || '|' || ".join(f'CAST("{c}" AS TEXT)' for c in pk_cols)
    pk_expr_pg = " || '|' || ".join(f'CAST("{c}" AS TEXT)' for c in pk_cols)

    sqlite_ids = {r[0] for r in sqlite_conn.execute(f'SELECT {pk_expr_sqlite} FROM "{table}"').fetchall()}
    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT {pk_expr_pg} FROM "{table}"')
        pg_ids = {r[0] for r in cur.fetchall()}

    only_sqlite = list(sqlite_ids - pg_ids)[:10]
    only_pg = list(pg_ids - sqlite_ids)[:10]
    if only_sqlite:
        log.error("    first 10 PKs present in SQLite but missing in PG: %s", only_sqlite)
    if only_pg:
        log.error("    first 10 PKs present in PG but missing in SQLite: %s", only_pg)


def _verify_sample_rows(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    table: str,
    sample_size: int = 100,
) -> bool:
    """Diff up to ``sample_size`` random rows column-by-column.
    Ignores generated columns (PG computes them). Returns False on any
    mismatch.
    """
    pk_cols = PRIMARY_KEYS.get(table)
    if not pk_cols:
        return True

    total = _count_sqlite(sqlite_conn, table)
    if total == 0:
        return True

    # SQLite ORDER BY RANDOM() scans the whole table; fine at our scale
    # (biggest table is ~200k rows). For multi-million-row tables you'd
    # want TABLESAMPLE-style sampling instead.
    k = min(sample_size, total)
    sqlite_conn.row_factory = sqlite3.Row
    sample_rows = sqlite_conn.execute(f'SELECT * FROM "{table}" ORDER BY RANDOM() LIMIT {k}').fetchall()

    copy_cols = _get_pg_copy_columns(pg_conn, table)
    compare_cols = [c for c in copy_cols if c in _get_sqlite_columns_set(sqlite_conn, table)]

    mismatches = 0
    with pg_conn.cursor() as cur:
        for srow in sample_rows:
            where_sql = " AND ".join(f'"{c}" = %s' for c in pk_cols)
            params = tuple(srow[c] for c in pk_cols)
            select_cols_sql = ", ".join('"' + c + '"' for c in compare_cols)
            cur.execute(
                f'SELECT {select_cols_sql} FROM "{table}" WHERE {where_sql}',
                params,
            )
            prow = cur.fetchone()
            if prow is None:
                log.error("    [sample] %s: PK %s missing in PG", table, params)
                mismatches += 1
                continue
            for i, col in enumerate(compare_cols):
                s_val = srow[col]
                p_val = prow[i]
                if not _values_equal(s_val, p_val):
                    log.error(
                        "    [sample] %s.%s pk=%s: sqlite=%r  pg=%r",
                        table,
                        col,
                        params,
                        _truncate_repr(s_val),
                        _truncate_repr(p_val),
                    )
                    mismatches += 1
                    break  # one mismatch per row is enough

    if mismatches == 0:
        log.info("  [verify] %s: %d sampled row(s) match", table, len(sample_rows))
        return True
    log.error("  [verify] %s: %d/%d sampled row(s) mismatched", table, mismatches, len(sample_rows))
    return False


def _get_sqlite_columns_set(sqlite_conn: sqlite3.Connection, table: str) -> set[str]:
    return set(_get_sqlite_columns(sqlite_conn, table))


def _values_equal(a: Any, b: Any) -> bool:
    """Equality that tolerates the SQLite/PG marshalling differences we
    actually see: bytes↔memoryview and int↔int-promoted-to-bigint both
    compare equal here.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (bytes, bytearray, memoryview)) or isinstance(b, (bytes, bytearray, memoryview)):
        return bytes(a) == bytes(b)
    return a == b


def _truncate_repr(v: Any, limit: int = 80) -> str:
    r = repr(v)
    if len(r) > limit:
        return r[:limit] + "..."
    return r


# ─────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────


@contextmanager
def _open_sqlite(path: str):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _open_pg(dsn: str):
    # autocommit=False: each table runs as its own transaction and we
    # commit explicitly after a successful COPY.
    conn = psycopg.connect(dsn, autocommit=False)
    try:
        yield conn
    finally:
        conn.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sqlite", required=True, help="path to SQLite DB file")
    p.add_argument("--dsn", required=True, help="Postgres DSN")
    p.add_argument(
        "--tables",
        default="",
        help="comma-separated subset of tables to migrate (default: all, in FK order)",
    )
    p.add_argument("--batch-size", type=int, default=5000, help="rows per SQLite fetchmany batch")
    p.add_argument("--verify", action="store_true", help="run count + sampled-row verification after copy")
    p.add_argument(
        "--truncate-first",
        action="store_true",
        help="TRUNCATE each target table (RESTART IDENTITY CASCADE) before COPY",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    return p.parse_args(argv)


def _resolve_tables(arg_value: str) -> list[str]:
    if not arg_value:
        return list(TABLES)
    requested = [t.strip() for t in arg_value.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TABLES]
    if unknown:
        raise SystemExit(f"unknown table(s): {unknown}  (known: {TABLES})")
    # Preserve FK-safe order even when the user lists them out of order.
    order = {t: i for i, t in enumerate(TABLES)}
    return sorted(requested, key=lambda t: order[t])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(0)

    tables = _resolve_tables(args.tables)
    log.info("migrating %d table(s): %s", len(tables), tables)

    any_verify_failed = False
    with _open_sqlite(args.sqlite) as sconn, _open_pg(args.dsn) as pconn:
        for table in tables:
            try:
                _migrate_table(
                    sconn,
                    pconn,
                    table,
                    batch_size=args.batch_size,
                    truncate_first=args.truncate_first,
                )
            except Exception:
                log.exception("aborting migration due to error on table %s", table)
                return 1

        if args.verify:
            log.info("─── verification pass ───")
            for table in tables:
                counts_ok = _verify_counts(sconn, pconn, table)
                samples_ok = _verify_sample_rows(sconn, pconn, table)
                if not (counts_ok and samples_ok):
                    any_verify_failed = True

    if any_verify_failed:
        log.error("VERIFICATION FAILED — see errors above")
        return 1
    log.info("migration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
