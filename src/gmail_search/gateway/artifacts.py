"""Run-bound byte publication. Never open, mount, or resolve a guest path.

Files live in a private trusted directory; the registry transaction commits
ownership and quotas. Browser adapters must derive owner from their session.
All content downloads as an attachment; preview conversion belongs in a sandbox.
The stream iterator is a trusted HTTP/relay adapter, never guest Python code.
"""
import asyncio
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import uuid
from urllib.parse import quote

from .registry import AccessDenied


@dataclass(frozen=True)
class Artifact:
    id: str
    filename: str
    size: int

    @property
    def download_headers(self):
        return {
            'Content-Type': 'application/octet-stream',
            'Content-Disposition': "attachment; filename*=UTF-8''" + quote(self.filename, safe=''),
            'Cache-Control': 'private, no-store',
            'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "sandbox; default-src 'none'",
        }


class ArtifactStore:
    def __init__(self, root, capabilities, *, max_object_bytes=20 * 1024**2,
                 max_owner_bytes=200 * 1024**2, max_owner_objects=1024, upload_seconds=30):
        self.root = Path(root)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
            raise AccessDenied()
        for value in (max_object_bytes, max_owner_bytes, max_owner_objects, upload_seconds):
            if type(value) is not int or value <= 0:
                raise ValueError('Positive resource limits required')
        self.capabilities, self.registry = capabilities, capabilities.registry
        self.max_object_bytes, self.max_owner_bytes = max_object_bytes, max_owner_bytes
        self.max_owner_objects, self.upload_seconds = max_owner_objects, upload_seconds
        with self.registry._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                run_id TEXT NOT NULL, filename TEXT NOT NULL, size INTEGER NOT NULL,
                digest TEXT, committed INTEGER NOT NULL DEFAULT 0, expires REAL NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS artifact_owner ON artifacts(owner_id)')

    def _sync_directory(self):
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _authorize(self, token):
        return self.capabilities.authorize(token, audience='artifact', operation='artifact.commit')

    async def publish(self, token, stream, *, filename):
        self._authorize(token)
        try:
            filename_bytes = filename.encode("utf-8") if type(filename) is str else b""
        except UnicodeError:
            raise AccessDenied() from None
        if (type(filename) is not str or not 1 <= len(filename_bytes) <= 200
                or filename in ('.', '..') or any(ord(c) < 32 or ord(c) == 127 or c in '/\\' for c in filename)):
            raise AccessDenied()
        object_id = uuid.uuid4().hex
        # Reserve worst-case upload bytes before reading any guest bytes. Pending
        # reservations also bound simultaneous uploads across service processes.
        with self.registry._transaction() as db:
            run = self.capabilities._authorize(db, token, 'artifact', 'artifact.commit')
            usage = db.execute('SELECT COALESCE(SUM(size),0), COUNT(*) FROM artifacts WHERE owner_id=?', (run['owner_id'],)).fetchone()
            available = min(self.max_object_bytes, self.max_owner_bytes - usage[0])
            if available <= 0 or usage[1] >= self.max_owner_objects:
                raise AccessDenied()
            expires = min(run['deadline'], self.registry.clock() + self.upload_seconds)
            db.execute('INSERT INTO artifacts(id,owner_id,conversation_id,run_id,filename,size,expires) VALUES(?,?,?,?,?,?,?)',
                       (object_id, run['owner_id'], run['conversation_id'], run['run_id'], filename, available, expires))
        path, fd, committed = self.root / object_id, None, False
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            size, digest = 0, hashlib.sha256()
            iterator = aiter(stream)
            async with asyncio.timeout(self.upload_seconds):
                while True:
                    pending = asyncio.ensure_future(anext(iterator))
                    try:
                        while not pending.done():
                            await asyncio.wait({pending}, timeout=0.1)
                            self._authorize(token)
                        try:
                            chunk = pending.result()
                        except StopAsyncIteration:
                            break
                    finally:
                        if not pending.done():
                            pending.cancel()
                            await asyncio.wait({pending}, timeout=0.1)
                            # A broken trusted adapter must not pin upload cleanup.
                            pending.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                    self._authorize(token)
                    if type(chunk) is not bytes or size + len(chunk) > available:
                        raise AccessDenied()
                    size += len(chunk)
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise AccessDenied()
                        view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = None
            self._sync_directory()
            with self.registry._transaction() as db:
                self.capabilities._authorize(db, token, 'artifact', 'artifact.commit')
                result = db.execute('UPDATE artifacts SET size=?,digest=?,committed=1 WHERE id=? AND committed=0 AND expires>?',
                                    (size, digest.hexdigest(), object_id, self.registry.clock()))
                if result.rowcount != 1:
                    raise AccessDenied()
            committed = True
            return Artifact(object_id, filename, size)
        except (OSError, TimeoutError, UnicodeError):
            raise AccessDenied() from None
        finally:
            if fd is not None:
                os.close(fd)
            if not committed:
                path.unlink(missing_ok=True)
                self._sync_directory()
                with self.registry._transaction() as db:
                    db.execute('DELETE FROM artifacts WHERE id=? AND committed=0', (object_id,))

    def _metadata(self, db, owner_id, conversation_id, object_id):
        self.registry._owner(owner_id)
        if type(object_id) is not str or not re.fullmatch('[a-f0-9]{32}', object_id):
            raise AccessDenied()
        row = db.execute('SELECT * FROM artifacts WHERE id=? AND owner_id=? AND conversation_id=? AND committed=1',
                         (object_id, owner_id, conversation_id)).fetchone()
        if not row:
            raise AccessDenied()
        return row

    def metadata(self, owner_id, conversation_id, object_id):
        with self.registry._transaction() as db:
            row = self._metadata(db, owner_id, conversation_id, object_id)
            return Artifact(row['id'], row['filename'], row['size'])

    def read(self, owner_id, conversation_id, object_id):
        with self.registry._transaction() as db:
            row = self._metadata(db, owner_id, conversation_id, object_id)
        try:
            fd = os.open(self.root / object_id, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as file:
                info = os.fstat(file.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size != row['size']
                        or info.st_size > self.max_object_bytes):
                    raise AccessDenied()
                data = file.read(row['size'] + 1)
            if len(data) != row['size'] or hashlib.sha256(data).hexdigest() != row['digest']:
                raise AccessDenied()
        except OSError:
            raise AccessDenied() from None
        self.registry._owner(owner_id)
        return data

    def reap_pending(self):
        """Trusted sweeper removes expired, interrupted uploads; never committed data."""
        with self.registry._transaction() as db:
            rows = db.execute('SELECT id FROM artifacts WHERE committed=0 AND expires<=?', (self.registry.clock(),)).fetchall()
            for row in rows:
                if not re.fullmatch('[a-f0-9]{32}', row['id']):
                    raise AccessDenied()
                (self.root / row['id']).unlink(missing_ok=True)
                self._sync_directory()
                db.execute('DELETE FROM artifacts WHERE id=? AND committed=0', (row['id'],))
            return len(rows)
