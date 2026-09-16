"""Trusted worker lifecycle foundation. This module does not launch production VMs.

Only trusted controllers may call this API; guest capabilities never reach it.
An injected backend owns a dedicated, persistent handle namespace. It MUST:
* launch only its fixed, verified image with no host mounts or general network;
* enforce every resource limit outside the guest, including an independent
  lease/deadline watchdog that survives controller loss;
* make launch/stop idempotent by the opaque handle, supervise the actual VMM
  and descendants, and return from stop only after resources are gone;
* provide bounded operations and inventory across controller/backend restarts.

No subprocess adapter is provided: jailer/cgroups, relay, watchdog and isolated
host qualification remain prerequisites. Synthetic tests prove lifecycle logic,
not kernel isolation. A trusted service must call reconcile periodically.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
import re
import stat
from typing import Protocol
import uuid

from .registry import AccessDenied, RunLease


@dataclass(frozen=True)
class WorkerLimits:
    vcpus: int = 1
    memory_mib: int = 512
    pids: int = 128
    disk_bytes: int = 1024**3
    output_bytes: int = 8 * 1024**2
    wall_seconds: int = 3600

    def __post_init__(self):
        ceilings = {'vcpus': 8, 'memory_mib': 16384, 'pids': 4096,
                    'disk_bytes': 32 * 1024**3, 'output_bytes': 256 * 1024**2,
                    'wall_seconds': 86400}
        for key, maximum in ceilings.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('Worker resource limits must be positive bounded integers')


class WorkerBackend(Protocol):
    """Trusted fixed-config backend; never construct one from guest arguments."""
    namespace: str

    def launch(self, handle: str, lease: RunLease, limits: WorkerLimits) -> None: ...
    def renew(self, handle: str, lease: RunLease) -> None: ...
    def stop(self, handle: str) -> None: ...
    def inventory(self) -> set[str]: ...


class WorkerController:
    def __init__(self, registry, backend: WorkerBackend, *, limits=WorkerLimits(),
                 max_workers=8, max_owner_workers=2):
        if (type(limits) is not WorkerLimits or type(backend.namespace) is not str
                or not re.fullmatch(r'[a-z][a-z0-9-]{0,63}', backend.namespace)):
            raise ValueError('Trusted backend namespace and fixed limits required')
        if any(type(n) is not int or not 1 <= n <= 1024 for n in (max_workers, max_owner_workers)):
            raise ValueError('Positive bounded worker quotas required')
        self.registry, self.backend, self.limits = registry, backend, limits
        self.max_workers, self.max_owner_workers = max_workers, max_owner_workers
        # Registry already requires its parent to be private and service-owned.
        self.lock_path = registry.path.parent / 'worker-controller.lock'
        with self._lock(), registry._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS workers (
                run_id TEXT PRIMARY KEY, handle TEXT UNIQUE NOT NULL,
                namespace TEXT NOT NULL, owner_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('launching','running','stopping','stopped')))''')

    @contextmanager
    def _lock(self):
        """Serialize external side effects across local controllers, not DB reads."""
        fd = None
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600):
                raise AccessDenied()
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        except OSError:
            raise AccessDenied() from None
        finally:
            if fd is not None:
                os.close(fd)

    def _live(self, db, run_id):
        row = self.registry._active(db, run_id)
        self.registry._fence(db, row)
        return self.registry._lease(row)

    def _inventory(self):
        try:
            handles = self.backend.inventory()
        except Exception:
            raise AccessDenied() from None
        if not isinstance(handles, (set, frozenset)) or any(
                type(handle) is not str or not re.fullmatch('[a-f0-9]{32}', handle) for handle in handles):
            raise AccessDenied()
        return handles

    def _stop(self, row):
        # Revoke first, including when stop itself fails or the controller dies.
        with self.registry._transaction() as db:
            self.registry._finish(db, row['run_id'], 'cancelled')
            db.execute("UPDATE workers SET state='stopping' WHERE run_id=?", (row['run_id'],))
        try:
            self.backend.stop(row['handle'])
            if row['handle'] in self._inventory():
                raise AccessDenied()
        except Exception:
            raise AccessDenied() from None
        with self.registry._transaction() as db:
            db.execute("UPDATE workers SET state='stopped' WHERE run_id=?", (row['run_id'],))

    def start(self, run_id: str) -> str:
        """Launch an existing trusted run once; no caller image, path or flags."""
        with self._lock():
            with self.registry._transaction() as db:
                lease = self._live(db, run_id)
                old = db.execute('SELECT * FROM workers WHERE run_id=?', (run_id,)).fetchone()
                if old:
                    if old['namespace'] != self.backend.namespace or old['state'] != 'running':
                        raise AccessDenied()
                else:
                    usage = db.execute("SELECT COUNT(*),COALESCE(SUM(owner_id=?),0) FROM workers WHERE state!='stopped'", (lease.owner_id,)).fetchone()
                    if (usage[0] >= self.max_workers or usage[1] >= self.max_owner_workers
                            or lease.deadline - self.registry.clock() > self.limits.wall_seconds):
                        raise AccessDenied()
                    handle = uuid.uuid4().hex
                    db.execute("INSERT INTO workers VALUES(?,?,?,?,'launching')", (run_id, handle, self.backend.namespace, lease.owner_id))
            if old:
                if old['handle'] not in self._inventory():
                    self._stop(old)
                    raise AccessDenied()
                return old['handle']
            try:
                self.backend.launch(handle, lease, self.limits)
                if handle not in self._inventory():
                    raise AccessDenied()
                with self.registry._transaction() as db:
                    self._live(db, run_id)  # Launch may race owner revocation/cancellation.
                    db.execute("UPDATE workers SET state='running' WHERE run_id=?", (run_id,))
                return handle
            except Exception:
                with self.registry._transaction() as db:
                    row = db.execute('SELECT * FROM workers WHERE run_id=?', (run_id,)).fetchone()
                self._stop(row)
                raise AccessDenied() from None

    def cancel(self, run_id: str) -> None:
        """Revoke immediately even when a launch/teardown holds the worker lock.

        Lock contention returns AccessDenied after revocation; the launching
        controller's final check or the next sweeper pass completes teardown.
        """
        with self.registry._transaction() as db:
            row = db.execute('SELECT * FROM workers WHERE run_id=?', (run_id,)).fetchone()
            if row and row['namespace'] != self.backend.namespace:
                raise AccessDenied()
            self.registry._finish(db, run_id, 'cancelled')
        with self._lock():
            with self.registry._transaction() as db:
                row = db.execute('SELECT * FROM workers WHERE run_id=?', (run_id,)).fetchone()
            if row and row['state'] != 'stopped':
                self._stop(row)

    def heartbeat(self, run_id: str) -> RunLease:
        """Trusted worker health extends an online lease, never the hard deadline."""
        with self._lock():
            with self.registry._transaction() as db:
                row = db.execute('SELECT * FROM workers WHERE run_id=?', (run_id,)).fetchone()
                if not row or row['namespace'] != self.backend.namespace or row['state'] != 'running':
                    raise AccessDenied()
            try:
                if row['handle'] not in self._inventory():
                    raise AccessDenied()
                lease = self.registry.heartbeat(run_id)
                self.backend.renew(row['handle'], lease)
                with self.registry._transaction() as db:
                    self._live(db, run_id)
                return lease
            except Exception:
                self._stop(row)
                raise AccessDenied() from None

    def reconcile(self) -> int:
        """Restart-safe sweeper for invalid runs, interrupted launches and orphans.

        Stop failures retain durable bindings/quota for the next pass. Continue
        independent cleanup, then fail closed so the controller can alert/retry.
        """
        with self._lock():
            handles = self._inventory()
            with self.registry._transaction() as db:
                rows = db.execute('SELECT * FROM workers WHERE namespace=?', (self.backend.namespace,)).fetchall()
                known = {row['handle'] for row in rows}
                stopping = []
                for row in rows:
                    if row['state'] == 'stopped':
                        if row['handle'] in handles:
                            stopping.append(row)
                        continue
                    valid = row['state'] == 'running' and row['handle'] in handles
                    if valid:
                        try:
                            self._live(db, row['run_id'])
                        except AccessDenied:
                            valid = False
                    if not valid:
                        stopping.append(row)
            removed, failed = 0, False
            for row in stopping:
                try:
                    self._stop(row)
                    removed += 1
                except AccessDenied:
                    failed = True
            for handle in handles - known:
                try:
                    self.backend.stop(handle)
                    if handle in self._inventory():
                        raise AccessDenied()
                    removed += 1
                except Exception:
                    failed = True
            if failed:
                raise AccessDenied()
            return removed
