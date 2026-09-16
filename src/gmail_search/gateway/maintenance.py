"""One-store, durable controller readiness; no PostgreSQL migration orchestration.

Only an administrator initializes/upgrades this file or supplies verification.
Runtime reads never create it. Old binaries and existing database credentials
still require deployment fencing; this module cannot terminate their work.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time

from .partition_profiles import PartitionSchemaProfile, require_profile
from .registry import AccessDenied, _SCHEMA


_TABLES = ('store_gate', 'store_releases')
_DIGEST = re.compile(r'[a-f0-9]{64}\Z')
_LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z')
_MAX_INT = 2**63 - 1
_GATE_SCHEMA = '''
CREATE TABLE store_releases (
 store_id TEXT NOT NULL, epoch INTEGER NOT NULL CHECK(epoch>0),
 profile TEXT NOT NULL, migration_id TEXT NOT NULL UNIQUE,
 owner_set_digest TEXT NOT NULL, procedure_digest TEXT NOT NULL,
 PRIMARY KEY(store_id,epoch));
CREATE TABLE store_gate (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1), store_id TEXT NOT NULL,
 active_epoch INTEGER NOT NULL CHECK(active_epoch>0),
 state TEXT NOT NULL CHECK(state IN ('MAINTENANCE','INDEX_PENDING','READY')),
 revision INTEGER NOT NULL CHECK(revision>0),
 phase1_digest TEXT, qualification_digest TEXT);
'''


@dataclass(frozen=True)
class ReleaseIdentity:
    store_id: str
    profile: PartitionSchemaProfile
    release_epoch: int

    def __post_init__(self):
        require_profile(self.profile)
        if (type(self.store_id) is not str or not _LABEL.fullmatch(self.store_id)
                or type(self.release_epoch) is not int
                or not 1 <= self.release_epoch <= _MAX_INT):
            raise ValueError('Invalid trusted release identity')


@dataclass(frozen=True)
class GateSnapshot:
    identity: ReleaseIdentity
    state: str
    revision: int
    migration_id: str
    owner_set_digest: str
    procedure_digest: str
    phase1_digest: str | None
    qualification_digest: str | None


def _duration(value, maximum):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 < value <= maximum):
        raise ValueError('Bounded trusted timeout required')
    return value


def _binding(migration_id, owner_set_digest, procedure_digest):
    if (type(migration_id) is not str or not _LABEL.fullmatch(migration_id)
            or any(type(value) is not str or not _DIGEST.fullmatch(value)
                for value in (owner_set_digest, procedure_digest))):
        raise AccessDenied()


def _validate_path(path, *, create=False):
    """Private local ancestors are trusted, as for Registry and SQLite journals."""
    path = Path(path).absolute()
    fd = None
    try:
        parent = path.parent.lstat()
        if (not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700
                or parent.st_uid != os.getuid()):
            raise AccessDenied()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            | (os.O_CREAT if create else 0), 0o600)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise AccessDenied()
        if create:
            # SQLite FULL syncs transaction contents. Also make the initial
            # database/lock directory entries durable before acknowledging init.
            # Repeat on administrative retries after a failed initial fsync.
            os.fsync(fd)
            parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except OSError:
        raise AccessDenied() from None
    finally:
        if fd is not None:
            os.close(fd)
    return path


@contextmanager
def _connection(path, *, write=False, create=False):
    path = _validate_path(path, create=create)
    db = None
    try:
        db = sqlite3.connect(path.as_uri() + ('?mode=rw' if write else '?mode=ro'),
            uri=True, timeout=2, isolation_level=None)
        db.row_factory = sqlite3.Row
        # This first profile deliberately supports SQLite's rollback journal.
        # It neither switches modes nor creates WAL/SHM siblings at runtime.
        if db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
            raise AccessDenied()
        if write:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
        yield db
        if write:
            db.commit()
    except sqlite3.Error:
        raise AccessDenied() from None
    finally:
        if db is not None:
            db.close()


def gate_enabled(db):
    # A partial gate schema also disables the legacy-default authority path.
    return (bool(db.execute("SELECT 1 FROM sqlite_master WHERE name IN (?,?) LIMIT 1", _TABLES).fetchone())
        or any(row[1] == 'release_epoch' for row in db.execute('PRAGMA table_info(runs)')))


def _snapshot(db):
    try:
        rows = db.execute('''SELECT g.*,r.profile,r.migration_id,
            r.owner_set_digest,r.procedure_digest FROM store_gate g
            JOIN store_releases r ON r.store_id=g.store_id AND r.epoch=g.active_epoch''').fetchall()
        if len(rows) != 1 or db.execute('SELECT COUNT(*) FROM store_gate').fetchone()[0] != 1:
            raise AccessDenied()
        row = rows[0]
        identity = ReleaseIdentity(row['store_id'], PartitionSchemaProfile(row['profile']), row['active_epoch'])
        _binding(row['migration_id'], row['owner_set_digest'], row['procedure_digest'])
        if (row['singleton'] != 1 or type(row['revision']) is not int
                or not 1 <= row['revision'] <= _MAX_INT
                or row['state'] not in ('MAINTENANCE', 'INDEX_PENDING', 'READY')):
            raise AccessDenied()
        for name, required in (('phase1_digest', row['state'] != 'MAINTENANCE'),
                ('qualification_digest', row['state'] == 'READY')):
            value = row[name]
            if ((required and (type(value) is not str or not _DIGEST.fullmatch(value)))
                    or (not required and value is not None)):
                raise AccessDenied()
        return GateSnapshot(identity, row['state'], row['revision'], row['migration_id'],
            row['owner_set_digest'], row['procedure_digest'], row['phase1_digest'],
            row['qualification_digest'])
    except (sqlite3.Error, ValueError, TypeError, KeyError):
        raise AccessDenied() from None


def require_ready_in(db, identity):
    """Use the caller's transaction, never recursively acquire a writer lock."""
    if identity is None:
        if gate_enabled(db):
            raise AccessDenied()
        return None
    if type(identity) is not ReleaseIdentity:
        raise AccessDenied()
    if db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
        raise AccessDenied()
    snapshot = _snapshot(db)
    if snapshot.identity != identity or snapshot.state != 'READY':
        raise AccessDenied()
    return snapshot


@contextmanager
def _publication_lock(path, *, exclusive, timeout, create=False):
    path = Path(path).absolute()
    fd = None
    try:
        lock_path = _validate_path(path.with_name(path.name + '.publication.lock'), create=create)
        fd = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        end = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= end:
                    raise AccessDenied() from None
                time.sleep(min(.01, max(0, end - time.monotonic())))
        yield
    except OSError:
        raise AccessDenied() from None
    finally:
        if fd is not None:
            # Never retry an uncertain fd number. Process teardown is the final
            # boundary if a local close cannot be acknowledged.
            os.close(fd)


class GateReader:
    def __init__(self, path, identity, *, lock_timeout=2):
        if type(identity) is not ReleaseIdentity:
            raise ValueError('Pinned trusted release identity required')
        self.path, self._identity = Path(path).absolute(), identity
        self.lock_timeout = _duration(lock_timeout, 2)

    @property
    def identity(self):
        return self._identity

    def require_ready(self):
        with _connection(self.path) as db:
            return require_ready_in(db, self.identity)

    @contextmanager
    def publication_guard(self):
        """Trusted bounded installer owns this guard through its final cleanup.

        No SQLite transaction spans external work. A successful close waits for
        all prior publishers; lock timeout is failure, not a closed gate ACK.
        """
        with _publication_lock(self.path, exclusive=False, timeout=self.lock_timeout):
            snapshot = self.require_ready()
            yield snapshot


class MaintenanceAdmin:
    """Administrator capability by composition; never expose this object to guests.

    verifier(snapshot, target_state, *, deadline) must perform the separately
    qualified external checks and return a lowercase SHA256 evidence digest.
    It must bound/drain its own work. No verifier is supplied by this module.
    """
    def __init__(self, path, *, verifier=None, lock_timeout=2, verification_timeout=30):
        if verifier is not None and not callable(verifier):
            raise ValueError('Trusted verifier must be callable')
        self.path = Path(path).absolute()
        self._verifier = verifier
        self.lock_timeout = _duration(lock_timeout, 2)
        self.verification_timeout = _duration(verification_timeout, 30)

    def status(self):
        with _connection(self.path) as db:
            return _snapshot(db)

    @staticmethod
    def _invalidate(db):
        db.execute("UPDATE runs SET status='cancelled' WHERE status='active'")
        db.execute('UPDATE capabilities SET revoked=1')
        db.execute('UPDATE conversations SET writer=NULL,fence=fence+1')

    @staticmethod
    def _release(db, identity, migration_id, owner_set_digest, procedure_digest):
        if type(identity) is not ReleaseIdentity:
            raise AccessDenied()
        _binding(migration_id, owner_set_digest, procedure_digest)
        db.execute('INSERT INTO store_releases VALUES(?,?,?,?,?,?)',
            (identity.store_id, identity.release_epoch, identity.profile.value,
                migration_id, owner_set_digest, procedure_digest))

    def initialize_closed(self, identity, *, migration_id, owner_set_digest, procedure_digest):
        """Explicitly initialize a new file or atomically upgrade an old Registry."""
        if type(identity) is not ReleaseIdentity:
            raise AccessDenied()
        _binding(migration_id, owner_set_digest, procedure_digest)
        with _publication_lock(self.path, exclusive=True, timeout=self.lock_timeout, create=True):
            with _connection(self.path, write=True, create=True) as db:
                if gate_enabled(db):
                    snapshot = _snapshot(db)
                    if (snapshot.identity != identity or snapshot.state != 'MAINTENANCE'
                            or snapshot.migration_id != migration_id
                            or snapshot.owner_set_digest != owner_set_digest
                            or snapshot.procedure_digest != procedure_digest):
                        raise AccessDenied()
                    return snapshot
                for statement in (_SCHEMA + _GATE_SCHEMA).split(';'):
                    if statement.strip():
                        db.execute(statement)
                db.execute('ALTER TABLE runs ADD COLUMN release_epoch INTEGER')
                self._release(db, identity, migration_id, owner_set_digest, procedure_digest)
                db.execute("INSERT INTO store_gate VALUES(1,?,?,'MAINTENANCE',1,NULL,NULL)",
                    (identity.store_id, identity.release_epoch))
                self._invalidate(db)
                return _snapshot(db)

    def begin_maintenance(self, identity, *, migration_id, owner_set_digest, procedure_digest, expected_revision):
        if type(identity) is not ReleaseIdentity or type(expected_revision) is not int:
            raise AccessDenied()
        _binding(migration_id, owner_set_digest, procedure_digest)
        with _publication_lock(self.path, exclusive=True, timeout=self.lock_timeout):
            with _connection(self.path, write=True) as db:
                current = _snapshot(db)
                if (current.state == 'MAINTENANCE' and current.revision == expected_revision + 1
                        and current.identity == identity and current.migration_id == migration_id
                        and current.owner_set_digest == owner_set_digest
                        and current.procedure_digest == procedure_digest):
                    return current
                if (current.state != 'READY' or current.revision != expected_revision
                        or identity.store_id != current.identity.store_id
                        or identity.release_epoch <= current.identity.release_epoch):
                    raise AccessDenied()
                self._release(db, identity, migration_id, owner_set_digest, procedure_digest)
                db.execute("UPDATE store_gate SET active_epoch=?,state='MAINTENANCE',revision=revision+1,phase1_digest=NULL,qualification_digest=NULL WHERE revision=?",
                    (identity.release_epoch, expected_revision))
                self._invalidate(db)
                return _snapshot(db)

    def record_index_pending(self, expected):
        return self._advance(expected, 'MAINTENANCE', 'INDEX_PENDING', 'phase1_digest')

    def publish_ready(self, expected):
        return self._advance(expected, 'INDEX_PENDING', 'READY', 'qualification_digest')

    def _advance(self, expected, source, target, field):
        if type(expected) is not GateSnapshot or expected.state != source or self._verifier is None:
            raise AccessDenied()
        with _publication_lock(self.path, exclusive=True, timeout=self.lock_timeout):
            current = self.status()
            if (current.state == target and current.revision == expected.revision + 1
                    and current.identity == expected.identity
                    and current.migration_id == expected.migration_id
                    and current.owner_set_digest == expected.owner_set_digest
                    and current.procedure_digest == expected.procedure_digest
                    and (source == 'MAINTENANCE' or current.phase1_digest == expected.phase1_digest)):
                return current
            if current != expected:
                raise AccessDenied()
            deadline = time.monotonic() + self.verification_timeout
            try:
                digest = self._verifier(current, target, deadline=deadline)
            except Exception:
                raise AccessDenied() from None
            if (time.monotonic() >= deadline or type(digest) is not str or not _DIGEST.fullmatch(digest)):
                raise AccessDenied()
            with _connection(self.path, write=True) as db:
                if _snapshot(db) != expected:
                    raise AccessDenied()
                # field is selected solely by the two fixed public methods.
                db.execute(f'UPDATE store_gate SET state=?,revision=revision+1,{field}=? WHERE revision=?',
                    (target, digest, expected.revision))
                return _snapshot(db)
