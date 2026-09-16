"""An apply may only touch the database it was declared against.

The migration tools refused every DSN that was not the disposable fixture,
which is correct and is also why they cannot run the migration they exist for.
Production is therefore a *second* declared target with its own conditions, not
a relaxed version of the first — a deleted check would have made every
rehearsal run one typo away from rewriting the live mailbox.

The condition that does the work is the cluster confirmation, and its third
field is the reason it works. `<database>:<system identifier>` alone is *not*
distinguishing: a physical copy — pg_basebackup, PITR, a snapshot — carries the
same system identifier, database name and OID as its source, so production and a
restore of production present identical tokens. The marker nonce has to be
written into the target, and a copy taken beforehand does not carry it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import time
import types

import pytest
from psycopg.pq import TransactionStatus

ROOT = Path(__file__).parents[1]

_FIXTURE_DB = "gms_owner_partitions_test_text_deadbeef"
_LIVE_DB = "gmail_search"
_LIVE_SYSTEM_ID = "7312995849283746510"
_NONCE = "a3f1" * 8          # 32 hex characters, the minimum the guard accepts
_MARKER = "gms-migration-target:" + _NONCE


@pytest.fixture(scope="module")
def migration():
    spec = importlib.util.spec_from_file_location(
        "text_partition_migration", ROOT / "deploy/public/migrate_text_owner_partitions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeConn:
    """Enough connection to exercise the target decision and nothing more.

    The guard runs before any migration work, so a real cluster would only slow
    the test down and would make the remote-host case untestable.
    """

    def __init__(self, *, host, hostaddr, port, dbname, system_id=_LIVE_SYSTEM_ID,
                 marker=_MARKER):
        self.info = types.SimpleNamespace(
            transaction_status=TransactionStatus.IDLE,
            host=host,
            hostaddr=hostaddr,
            port=port,
            dbname=dbname,
        )
        self._system_id = system_id
        self._marker = marker

    def execute(self, _sql):
        row = (self.info.dbname, self._system_id, self._marker)
        return types.SimpleNamespace(fetchone=lambda: row)


def _fixture_conn():
    return _FakeConn(host="127.0.0.1", hostaddr="127.0.0.1", port=55440, dbname=_FIXTURE_DB)


def _live_conn():
    return _FakeConn(host="127.0.0.1", hostaddr="127.0.0.1", port=5544, dbname=_LIVE_DB)


def _confirmation(conn):
    return f"{conn.info.dbname}:{_LIVE_SYSTEM_ID}:{_NONCE}"


@pytest.fixture
def production(monkeypatch, migration, tmp_path):
    """Declare production with a backup that exists. Confirmation is per-test."""
    backup = tmp_path / "backup-receipt"
    backup.write_text("verified\n")
    monkeypatch.setenv(migration.TARGET_ENV, "production")
    monkeypatch.setenv(migration.BACKUP_ENV, str(backup))
    return backup


# ── the default target ───────────────────────────────────────────────────────

def test_default_target_is_the_disposable_fixture(monkeypatch, migration):
    monkeypatch.delenv(migration.TARGET_ENV, raising=False)
    migration._require(_fixture_conn(), apply=True)


def test_default_target_refuses_the_live_database(monkeypatch, migration):
    monkeypatch.delenv(migration.TARGET_ENV, raising=False)
    with pytest.raises(ValueError, match="disposable"):
        migration._require(_live_conn(), apply=True)


def test_an_unknown_target_is_refused(monkeypatch, migration):
    monkeypatch.setenv(migration.TARGET_ENV, "staging")
    with pytest.raises(ValueError, match="not a known apply target"):
        migration._require(_live_conn(), apply=True)


def test_a_read_only_call_needs_no_target(monkeypatch, migration):
    """`apply=False` inspects; only an apply is gated on where it points."""
    monkeypatch.delenv(migration.TARGET_ENV, raising=False)
    migration._require(_live_conn(), apply=False)


# ── the production target ────────────────────────────────────────────────────

def test_production_accepts_the_confirmed_cluster(migration, production, monkeypatch):
    conn = _live_conn()
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    migration._require(conn, apply=True)


def test_production_without_a_confirmation_is_refused(migration, production, monkeypatch):
    monkeypatch.delenv(migration.CONFIRM_ENV, raising=False)
    with pytest.raises(ValueError, match="CONFIRM"):
        migration._require(_live_conn(), apply=True)


def test_a_confirmation_for_another_cluster_is_refused(migration, production, monkeypatch):
    """A token minted against a different `initdb` cannot admit this one."""
    monkeypatch.setenv(migration.CONFIRM_ENV, f"{_LIVE_DB}:9999999999999999999:{_NONCE}")
    with pytest.raises(ValueError, match="match this cluster"):
        migration._require(_live_conn(), apply=True)


def test_a_physical_restore_cannot_lend_its_token_to_production(migration, production, monkeypatch):
    """The finding this nonce exists for, and the reason the first two fields
    are not enough.

    `system_identifier` is issued at initdb and copied verbatim by every
    physical copy — pg_basebackup, PITR, a filesystem snapshot. Database names
    and OIDs survive too. Measured on this project's own cold backup, live and
    backup both report 7630921061527392295. So an operator holding both a
    restore and production — the normal state right after taking the backup this
    tool insists on — has two clusters whose `<database>:<system identifier>` is
    byte-identical, and could read the token off one and apply it to the other.

    Only the marker distinguishes them: a copy taken before the marker was
    written does not carry it.
    """
    restore = _FakeConn(host="127.0.0.1", hostaddr="127.0.0.1", port=6432, dbname=_LIVE_DB)
    production_without_marker = _FakeConn(
        host="127.0.0.1", hostaddr="127.0.0.1", port=6433, dbname=_LIVE_DB, marker=None)
    # Same database name, same system identifier — nothing but the marker differs.
    assert restore.info.dbname == production_without_marker.info.dbname
    assert restore._system_id == production_without_marker._system_id
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(restore))
    with pytest.raises(ValueError, match="migration marker"):
        migration._require(production_without_marker, apply=True)


def test_an_unmarked_target_is_refused(migration, production, monkeypatch):
    conn = _FakeConn(host="127.0.0.1", hostaddr="127.0.0.1", port=5544,
                     dbname=_LIVE_DB, marker=None)
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    with pytest.raises(ValueError, match="migration marker"):
        migration._require(conn, apply=True)


@pytest.mark.parametrize("marker", [
    "gms-migration-target:short",
    "gms-migration-target:" + "z" * 40,
    "something-else:" + _NONCE,
    "",
])
def test_a_weak_or_wrong_marker_is_refused(migration, production, monkeypatch, marker):
    """Too short, not hex, wrong prefix, or empty — each refused, so the marker
    cannot be satisfied by a value someone types out of habit."""
    conn = _FakeConn(host="127.0.0.1", hostaddr="127.0.0.1", port=5544,
                     dbname=_LIVE_DB, marker=marker)
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    with pytest.raises(ValueError, match="migration marker|hex characters"):
        migration._require(conn, apply=True)


def test_a_confirmation_for_another_database_is_refused(migration, production, monkeypatch):
    """Right cluster, wrong database — the token names all three."""
    conn = _live_conn()
    monkeypatch.setenv(migration.CONFIRM_ENV, f"some_other_db:{_LIVE_SYSTEM_ID}:{_NONCE}")
    with pytest.raises(ValueError, match="match this cluster"):
        migration._require(conn, apply=True)


def test_production_requires_a_named_backup(migration, monkeypatch):
    conn = _live_conn()
    monkeypatch.setenv(migration.TARGET_ENV, "production")
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    monkeypatch.delenv(migration.BACKUP_ENV, raising=False)
    with pytest.raises(ValueError, match="BACKUP"):
        migration._require(conn, apply=True)


def test_an_empty_backup_path_is_refused(migration, monkeypatch, tmp_path):
    """The check is a declaration, not a verification — but an empty file is not
    even a declaration."""
    conn = _live_conn()
    empty = tmp_path / "empty-receipt"
    empty.touch()
    monkeypatch.setenv(migration.TARGET_ENV, "production")
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    monkeypatch.setenv(migration.BACKUP_ENV, str(empty))
    with pytest.raises(ValueError, match="empty"):
        migration._require(conn, apply=True)


def test_a_backup_path_that_does_not_exist_is_refused(migration, monkeypatch, tmp_path):
    conn = _live_conn()
    monkeypatch.setenv(migration.TARGET_ENV, "production")
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    monkeypatch.setenv(migration.BACKUP_ENV, str(tmp_path / "never-made"))
    with pytest.raises(ValueError, match="BACKUP"):
        migration._require(conn, apply=True)


def _data_directory(root, system_id, *, age_seconds=0):
    """A minimal stand-in for a cold copy: only `global/pg_control` is read."""
    control = root / "global" / "pg_control"
    control.parent.mkdir(parents=True, exist_ok=True)
    control.write_bytes(struct.pack("<Q", int(system_id)) + b"\0" * 8)
    if age_seconds:
        stamp = time.time() - age_seconds
        import os as _os
        _os.utime(control, (stamp, stamp))
    return root


def test_a_backup_of_this_cluster_is_accepted(migration, monkeypatch, tmp_path):
    """A cold copy carries its cluster identity in `global/pg_control`, so for
    the shape this project actually rehearsed the check is real verification and
    not a declaration."""
    conn = _live_conn()
    monkeypatch.setenv(migration.TARGET_ENV, "production")
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    monkeypatch.setenv(migration.BACKUP_ENV,
                       str(_data_directory(tmp_path / "backup", _LIVE_SYSTEM_ID)))
    migration._require(conn, apply=True)


def test_a_backup_of_a_different_cluster_is_refused(migration, monkeypatch, tmp_path):
    """The failure this catches: a backup left over from a different cluster, or
    from a previous incarnation of this one, restores nothing."""
    conn = _live_conn()
    monkeypatch.setenv(migration.TARGET_ENV, "production")
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    monkeypatch.setenv(migration.BACKUP_ENV,
                       str(_data_directory(tmp_path / "other", "1234567890123456789")))
    with pytest.raises(ValueError, match="backup of cluster 1234567890123456789"):
        migration._require(conn, apply=True)


def test_a_stale_backup_is_refused(migration, monkeypatch, tmp_path):
    """Right cluster, but taken during a previous attempt."""
    conn = _live_conn()
    monkeypatch.setenv(migration.TARGET_ENV, "production")
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    stale = _data_directory(tmp_path / "stale", _LIVE_SYSTEM_ID, age_seconds=90 * 3600)
    monkeypatch.setenv(migration.BACKUP_ENV, str(stale))
    with pytest.raises(ValueError, match="past the 48h bound"):
        migration._require(conn, apply=True)


def test_pg_control_layout_matches_postgres(migration, tmp_path):
    """Pins the offset the reader depends on: `system_identifier` is the first
    field of ControlFileData, a little-endian uint64 at offset 0. Verified
    against this project's own cold backup, which reports 7630921061527392295
    from both this reader and `pg_controldata`."""
    root = _data_directory(tmp_path / "d", "7630921061527392295")
    assert migration._backup_system_identifier(root) == "7630921061527392295"
    assert migration._backup_system_identifier(tmp_path / "not-a-datadir") is None


@pytest.mark.parametrize(
    ("host", "hostaddr"),
    [("db.internal", "10.0.0.7"), ("198.51.100.4", "198.51.100.4")],
)
def test_production_refuses_a_remote_target(migration, production, monkeypatch, host, hostaddr):
    """This tool takes ACCESS EXCLUSIVE locks and rewrites partitions. A DSN
    pointing off-box is the one mistake with no undo, so it is refused before
    the confirmation is even considered."""
    conn = _FakeConn(host=host, hostaddr=hostaddr, port=5432, dbname=_LIVE_DB)
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    with pytest.raises(ValueError, match="loopback or local-socket"):
        migration._require(conn, apply=True)


def test_a_unix_socket_counts_as_local(migration, production, monkeypatch):
    conn = _FakeConn(host="/var/run/postgresql", hostaddr="", port=5432, dbname=_LIVE_DB)
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    migration._require(conn, apply=True)


def test_production_still_requires_an_idle_connection(migration, production, monkeypatch):
    """Production relaxes *which database*, nothing else."""
    conn = _live_conn()
    conn.info.transaction_status = TransactionStatus.INTRANS
    monkeypatch.setenv(migration.CONFIRM_ENV, _confirmation(conn))
    with pytest.raises(ValueError, match="idle"):
        migration._require(conn, apply=True)
