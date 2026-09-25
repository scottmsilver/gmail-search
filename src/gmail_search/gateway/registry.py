"""Durable trusted operational metadata, never a mailbox database.

Only trusted controllers call these methods with server-owned identities. Guests
use capability authorization. SQLite serializes mutations across processes; keep
the database in a private local directory, not a network filesystem. Outstanding
spend remains reserved after cancellation until trusted provider settlement.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import sys
import math
import os
from pathlib import Path
import sqlite3
import stat
import time
import uuid


logger = logging.getLogger(__name__)
# Settlement logs once when spending first crosses each of these fractions.
BUDGET_WARNING_FRACTIONS = (0.8, 0.95)


SLOW_TRANSACTION_SECONDS = 0.5
# How long a transaction waits for the lock before failing closed. At 2 s,
# waits of ~1.3 s behind other commits stacked past it under parallel
# subagents and refused live runs' calls (2026-09-24).
BUSY_TIMEOUT_SECONDS = 10


def _log_slow_transaction(opened, locked, read_only):
    """Name the caller of a transaction that waited for or held the lock long.
    Writers wait at most the 2 s busy timeout, then fail as AccessDenied."""
    now = time.monotonic()
    if now - opened < SLOW_TRANSACTION_SECONDS:
        return
    caller = sys._getframe(3).f_code.co_name
    logger.warning('registry %s transaction in %s: waited %.0fms, held %.0fms', 'read' if read_only else 'write',
        caller, (locked - opened) * 1000, (now - locked) * 1000)


class AccessDenied(PermissionError):
    def __init__(self):
        super().__init__('Run authorization or resource reservation is unavailable.')


class BudgetExhausted(AccessDenied):
    """A reservation would exceed the owner's lifetime token ceiling."""
    def __init__(self, ceiling, spent, reserved, requested):
        PermissionError.__init__(self, f'Token budget exhausted: spent {spent:,} of {ceiling:,} tokens '
            f'({reserved:,} reserved in flight); this call needs {requested:,}.')
        self.ceiling, self.spent, self.reserved, self.requested = ceiling, spent, reserved, requested


def _budget_exhausted(db, budget_id, requested):
    row = db.execute('SELECT owner_id,ceiling,spent,reserved FROM budgets WHERE budget_id=?', (budget_id,)).fetchone()
    if row is None:
        return AccessDenied()
    logger.warning('token budget exhausted owner=%s ceiling=%d spent=%d reserved=%d requested=%d',
        row['owner_id'], row['ceiling'], row['spent'], row['reserved'], requested)
    return BudgetExhausted(row['ceiling'], row['spent'], row['reserved'], requested)


def _log_budget_threshold(db, budget_id, charged):
    row = db.execute('SELECT owner_id,ceiling,spent FROM budgets WHERE budget_id=?', (budget_id,)).fetchone()
    for fraction in BUDGET_WARNING_FRACTIONS:
        limit = row['ceiling'] * fraction
        if row['spent'] - charged < limit <= row['spent']:
            logger.warning('token budget %d%% used owner=%s spent=%d ceiling=%d',
                int(fraction * 100), row['owner_id'], row['spent'], row['ceiling'])


@dataclass(frozen=True)
class RunLease:
    run_id: str
    owner_id: str
    conversation_id: str
    fence: int
    workspace_version: int
    deadline: float
    lease_expires: float
    writer: bool
    budget_id: str | None


@dataclass(frozen=True)
class Reservation:
    run_id: str
    request_key: str
    units: int
    charged: int | None
    created: bool  # Only the first claimant may initiate the billable operation.


_SCHEMA = '''
CREATE TABLE IF NOT EXISTS conversations (
 owner_id TEXT, conversation_id TEXT, fence INTEGER NOT NULL DEFAULT 0,
 version INTEGER NOT NULL DEFAULT 0, writer TEXT,
 PRIMARY KEY(owner_id, conversation_id));
CREATE TABLE IF NOT EXISTS budgets (
 budget_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, ceiling INTEGER NOT NULL,
 reserved INTEGER NOT NULL DEFAULT 0, spent INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS runs (
 run_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
 request_key TEXT NOT NULL, fence INTEGER NOT NULL, workspace_version INTEGER NOT NULL,
 deadline REAL NOT NULL, lease_expires REAL NOT NULL, writer INTEGER NOT NULL,
 budget_id TEXT, status TEXT NOT NULL DEFAULT 'active',
 UNIQUE(owner_id, conversation_id, request_key));
CREATE TABLE IF NOT EXISTS reservations (
 run_id TEXT, request_key TEXT, units INTEGER NOT NULL, charged INTEGER,
 PRIMARY KEY(run_id, request_key));
CREATE TABLE IF NOT EXISTS capabilities (
 token_hash TEXT PRIMARY KEY, run_id TEXT NOT NULL, audience TEXT NOT NULL,
 operations TEXT NOT NULL, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
'''


def _label(value):
    if not isinstance(value, str) or not value or len(value) > 512 or '\x00' in value:
        raise AccessDenied()


def _ttl(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 86400:
        raise AccessDenied()
    return value


def _units(value):
    if type(value) is not int or not 0 <= value <= 10**12:
        raise AccessDenied()


class Registry:
    def __init__(self, path, *, is_active, clock=time.time, release_identity=None):
        self.path, self.is_active, self.clock = Path(path), is_active, clock
        self._release_identity = release_identity
        from .maintenance import ReleaseIdentity, _connection, _snapshot, gate_enabled
        if release_identity is not None:
            if type(release_identity) is not ReleaseIdentity:
                raise ValueError('Pinned trusted release identity required')
            # Administrative initialization/upgrading is separate. This runtime
            # path creates neither a missing file nor any schema/journal object.
            with _connection(self.path) as db:
                if _snapshot(db).identity != release_identity:
                    raise AccessDenied()
                if 'release_epoch' not in {row[1] for row in db.execute('PRAGMA table_info(runs)')}:
                    raise AccessDenied()
            return
        # SQLite reopens by pathname and may create journal siblings. The
        # containing directory must be private to this trusted service account.
        # Its ancestors remain part of the trusted deployment boundary.
        parent = self.path.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700 or parent.st_uid != os.getuid():
            raise AccessDenied()
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
                raise AccessDenied()
        finally:
            os.close(fd)
        with self._transaction() as db:
            if gate_enabled(db):
                # Updated legacy-default callers may still perform cleanup, but
                # cannot create authority or silently initialize a gated file.
                _snapshot(db)
                return
            # Run individual statements: executescript implicitly commits.
            for statement in _SCHEMA.split(';'):
                if statement.strip():
                    db.execute(statement)

    @contextmanager
    def _transaction(self, *, read_only=False):
        """One registry transaction. Writers take the write lock up front
        (BEGIN IMMEDIATE). A read-only one opens the file read-only and holds
        only a shared lock, so concurrent checks do not queue on the writer
        lock: capability checks every 50 ms per model stream made writers time
        out ("database is locked") once a run had parallel subagents (2026-09-24)."""
        db = None
        opened = time.monotonic()
        try:
            db = sqlite3.connect(self.path.absolute().as_uri() + ('?mode=ro' if read_only else '?mode=rw'),
                uri=True, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
            db.row_factory = sqlite3.Row
            if not read_only:
                db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN DEFERRED' if read_only else 'BEGIN IMMEDIATE')
            locked = time.monotonic()
            yield db
            db.commit()
            _log_slow_transaction(opened, locked, read_only)
        except sqlite3.Error as error:
            # SQLite's own message (e.g. "database is locked"); never row data.
            logger.warning('registry transaction refused: %s: %s', type(error).__name__, error)
            raise AccessDenied() from None
        finally:
            if db is not None:
                db.close()  # Any exception rolls back the uncommitted transaction.

    def _owner(self, owner_id):
        try:
            active = self.is_active(owner_id)
        except Exception:
            raise AccessDenied() from None
        if active is not True:
            raise AccessDenied()

    def _active(self, db, run_id):
        from .maintenance import require_ready_in
        require_ready_in(db, self._release_identity)
        run = db.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()
        now = self.clock()
        if not run or run['status'] != 'active' or min(run['deadline'], run['lease_expires']) <= now:
            raise AccessDenied()
        if self._release_identity is not None and run['release_epoch'] != self._release_identity.release_epoch:
            raise AccessDenied()
        self._owner(run['owner_id'])
        return run

    @staticmethod
    def _lease(run):
        return RunLease(**{name: bool(run[name]) if name == 'writer' else run[name] for name in RunLease.__dataclass_fields__})

    @staticmethod
    def _fence(db, run, *, writer=False):
        current = db.execute('SELECT * FROM conversations WHERE owner_id=? AND conversation_id=?', (run['owner_id'], run['conversation_id'])).fetchone()
        if current['fence'] != run['fence'] or current['version'] != run['workspace_version'] or (writer and (not run['writer'] or current['writer'] != run['run_id'])):
            raise AccessDenied()

    def create_budget(self, owner_id, ceiling):
        _label(owner_id)
        _units(ceiling)
        with self._transaction() as db:
            from .maintenance import require_ready_in
            require_ready_in(db, self._release_identity)
            self._owner(owner_id)
            budget_id = uuid.uuid4().hex
            db.execute('INSERT INTO budgets(budget_id,owner_id,ceiling) VALUES(?,?,?)', (budget_id, owner_id, ceiling))
            return budget_id

    def start_run(self, owner_id, conversation_id, *, request_key, writer=True, budget_id=None, lease_ttl=60, deadline_ttl=3600):
        for label in (owner_id, conversation_id, request_key):
            _label(label)
        _ttl(lease_ttl)
        _ttl(deadline_ttl)
        if type(writer) is not bool:
            raise AccessDenied()
        with self._transaction() as db:
            from .maintenance import require_ready_in
            require_ready_in(db, self._release_identity)
            self._owner(owner_id)
            old = db.execute('SELECT * FROM runs WHERE owner_id=? AND conversation_id=? AND request_key=?', (owner_id, conversation_id, request_key)).fetchone()
            if old:
                if old['writer'] != writer or old['budget_id'] != budget_id:
                    raise AccessDenied()
                return self._lease(self._active(db, old['run_id']))
            if budget_id and not db.execute('SELECT 1 FROM budgets WHERE budget_id=? AND owner_id=?', (budget_id, owner_id)).fetchone():
                raise AccessDenied()
            db.execute('INSERT OR IGNORE INTO conversations(owner_id,conversation_id) VALUES(?,?)', (owner_id, conversation_id))
            current = db.execute('SELECT * FROM conversations WHERE owner_id=? AND conversation_id=?', (owner_id, conversation_id)).fetchone()
            now, run_id = self.clock(), uuid.uuid4().hex
            if writer and current['writer']:
                incumbent = db.execute('SELECT * FROM runs WHERE run_id=?', (current['writer'],)).fetchone()
                if incumbent['status'] == 'active' and min(incumbent['deadline'], incumbent['lease_expires']) > now:
                    raise AccessDenied()
            fence = current['fence'] + int(writer)
            if writer:
                db.execute('UPDATE conversations SET fence=?,writer=? WHERE owner_id=? AND conversation_id=?', (fence, run_id, owner_id, conversation_id))
            deadline = now + deadline_ttl
            db.execute('INSERT INTO runs(run_id,owner_id,conversation_id,request_key,fence,workspace_version,deadline,lease_expires,writer,budget_id) VALUES(?,?,?,?,?,?,?,?,?,?)', (run_id, owner_id, conversation_id, request_key, fence, current['version'], deadline, min(now + lease_ttl, deadline), writer, budget_id))
            if self._release_identity is not None:
                db.execute('UPDATE runs SET release_epoch=? WHERE run_id=?',
                    (self._release_identity.release_epoch, run_id))
            return self._lease(self._active(db, run_id))

    def heartbeat(self, run_id, *, ttl=60):
        _ttl(ttl)
        with self._transaction() as db:
            run = self._active(db, run_id)
            self._fence(db, run)
            db.execute('UPDATE runs SET lease_expires=? WHERE run_id=?', (min(self.clock() + ttl, run['deadline']), run_id))
            return self._lease(self._active(db, run_id))

    @staticmethod
    def _finish(db, run_id, status):
        db.execute("UPDATE runs SET status=? WHERE run_id=? AND status='active'", (status, run_id))
        db.execute('UPDATE capabilities SET revoked=1 WHERE run_id=?', (run_id,))
        db.execute('UPDATE conversations SET writer=NULL WHERE writer=?', (run_id,))

    def cancel(self, run_id):
        with self._transaction() as db:
            self._finish(db, run_id, 'cancelled')

    def finish(self, run_id, *, status):
        """Finish without promoting workspace data; workspace commit is separate."""
        if status not in ('completed', 'failed'):
            raise AccessDenied()
        with self._transaction() as db:
            self._active(db, run_id)
            self._finish(db, run_id, status)

    def cancel_owner(self, owner_id):
        with self._transaction() as db:
            runs = db.execute("SELECT run_id FROM runs WHERE owner_id=? AND status='active'", (owner_id,)).fetchall()
            for run in runs:
                self._finish(db, run['run_id'], 'cancelled')
            return tuple(run['run_id'] for run in runs)

    def expire(self):
        """Return expired IDs for an external controller to terminate their workers."""
        with self._transaction() as db:
            now = self.clock()
            runs = db.execute("SELECT run_id FROM runs WHERE status='active' AND (deadline<=? OR lease_expires<=?)", (now, now)).fetchall()
            for run in runs:
                self._finish(db, run['run_id'], 'expired')
            return tuple(run['run_id'] for run in runs)

    def reserve(self, run_id, request_key, units):
        """Claim a billable call once; retries return created=False, never a new grant."""
        _label(request_key)
        _units(units)
        with self._transaction() as db:
            run = self._active(db, run_id)
            old = db.execute('SELECT * FROM reservations WHERE run_id=? AND request_key=?', (run_id, request_key)).fetchone()
            if old:
                if old['units'] != units:
                    raise AccessDenied()
                return Reservation(run_id, request_key, units, old['charged'], False)
            changed = db.execute('UPDATE budgets SET reserved=reserved+? WHERE budget_id=? AND owner_id=? AND reserved+spent+?<=ceiling', (units, run['budget_id'], run['owner_id'], units)).rowcount
            if changed != 1:
                raise _budget_exhausted(db, run['budget_id'], units)
            db.execute('INSERT INTO reservations(run_id,request_key,units) VALUES(?,?,?)', (run_id, request_key, units))
            return Reservation(run_id, request_key, units, None, True)

    def settle(self, run_id, request_key, charged):
        """Trusted billing completion, including after revocation; never guest input."""
        _units(charged)
        with self._transaction() as db:
            old = db.execute('SELECT r.*,u.budget_id FROM reservations r JOIN runs u USING(run_id) WHERE r.run_id=? AND r.request_key=?', (run_id, request_key)).fetchone()
            if not old or charged > old['units'] or (old['charged'] is not None and charged != old['charged']):
                logger.warning('settle refused run=%s: %s', run_id[:8], 'no reservation' if not old else
                    f"charged {charged} > reserved {old['units']}" if charged > old['units'] else 'charge changed')
                raise AccessDenied()
            if old['charged'] is None:
                db.execute('UPDATE budgets SET reserved=reserved-?,spent=spent+? WHERE budget_id=?', (old['units'], charged, old['budget_id']))
                db.execute('UPDATE reservations SET charged=? WHERE run_id=? AND request_key=?', (charged, run_id, request_key))
                _log_budget_threshold(db, old['budget_id'], charged)
