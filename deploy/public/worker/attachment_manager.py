#!/usr/bin/env python3
"""Private worker RPC manager. Root owns VM lifecycle and durable tombstones.

The only transport is a fixed Unix socket. The entrypoint retains the existing
synthetic-worker boundary guard; this is not production deployment enablement.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/opt/gmail-worker')
import attachment_rpc as rpc

LEASE_SECONDS = 3
MAX_RECORDS = 10000


def _private(path, mode, directory=False):
    info = path.lstat()
    if (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != mode
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)):
        raise RuntimeError('Unsafe manager state')


class Manager:
    def __init__(self, state_dir, backend, *, controller_uid):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
        _private(self.state_dir, 0o700, directory=True)
        self.backend, self.controller_uid = backend, controller_uid
        self.lock = threading.RLock()
        self.jobs = {}
        self.db = None
        self.lock_fd = None
        try:
            lock_path = self.state_dir / 'manager.lock'
            self.lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            _private(lock_path, 0o600)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.state_dir / 'jobs.sqlite'
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            os.close(fd)
            _private(path, 0o600)
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, controller INTEGER NOT NULL, context TEXT NOT NULL, binding TEXT, state TEXT NOT NULL, seq INTEGER NOT NULL DEFAULT 0)')
            self.db.commit()
            # Reconciliation failure aborts startup while preserving durable rows.
            self.backend.reconcile()
            with self.db:
                self.db.execute("UPDATE jobs SET state='stopped'")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    @staticmethod
    def _reply(h, status, code='ok', payload=b''):
        response = rpc.response_header(h, status, code=code, payload=payload)
        rpc.encode_frame(response, payload, response=True)
        return response, payload

    def _row(self, job_id):
        return self.db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()

    def _insert(self, h, binding, state):
        if self.db.execute('SELECT count(*) FROM jobs').fetchone()[0] >= MAX_RECORDS:
            return False
        with self.db:
            self.db.execute('INSERT INTO jobs(job_id,controller,context,binding,state) VALUES(?,?,?,?,?)',
                            (h['job_id'], self.controller_uid, h['context_sha256'], binding, state))
        return True

    def _stop(self, job_id):
        with self.db:
            self.db.execute("UPDATE jobs SET state='stopping' WHERE job_id=?", (job_id,))
        entry = self.jobs.get(job_id)
        try:
            if entry is not None and entry['job'] is not None:
                entry['job'].stop()
            else:
                # A start exception may have left a VM without a returned handle.
                self.backend.reconcile()
        except Exception:
            return False
        with self.db:
            self.db.execute("UPDATE jobs SET state='stopped' WHERE job_id=?", (job_id,))
        self.jobs.pop(job_id, None)
        return True

    def tick(self, *, now=None):
        with self.lock:
            now = time.monotonic() if now is None else now
            for job_id, entry in tuple(self.jobs.items()):
                if now >= min(entry['lease'], entry['deadline']) or self._row(job_id)['state'] == 'stopping':
                    self._stop(job_id)

    def _status(self, h, entry):
        job = entry['job']
        if job is None or job.error is not None:
            return self._reply(h, 'error', 'job_failed')
        if job.result is not None:
            try:
                return self._reply(h, 'done', payload=job.result)
            except rpc.ProtocolError:
                self._stop(h['job_id'])
                return self._reply(h, 'error', 'output_limit')
        return self._reply(h, 'running')

    def handle(self, h, payload, *, peer_uid):
        rpc.validate_request(h, payload)
        with self.lock:
            if type(peer_uid) is not int or peer_uid != self.controller_uid:
                return self._reply(h, 'error', 'denied')
            self.tick()
            row = self._row(h['job_id'])
            if row is not None and (row['controller'] != peer_uid or row['context'] != h['context_sha256']):
                return self._reply(h, 'error', 'binding_mismatch')
            op = h['op']
            if op == 'stop':
                if row is None:
                    if not self._insert(h, None, 'stopped'):
                        return self._reply(h, 'error', 'metadata_full')
                    return self._reply(h, 'stopped')
                if row['state'] == 'stopped' or self._stop(h['job_id']):
                    return self._reply(h, 'stopped')
                return self._reply(h, 'error', 'stop_pending')
            if row is not None and row['state'] == 'stopped':
                return self._reply(h, 'error', 'stopped')
            if op == 'start':
                binding = hashlib.sha256(json.dumps({k: v for k, v in h.items() if k != 'request_id'},
                                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                if row is not None:
                    if row['binding'] != binding:
                        return self._reply(h, 'error', 'binding_mismatch')
                    if row['state'] == 'stopping':
                        return self._reply(h, 'error', 'stop_pending')
                    return self._status(h, self.jobs[h['job_id']])
                if self.db.execute("SELECT 1 FROM jobs WHERE state!='stopped' LIMIT 1").fetchone():
                    return self._reply(h, 'error', 'busy')
                deadline = min(h['context']['deadline'], time.time() + 45)
                remaining = deadline - time.time()
                if remaining <= 0:
                    return self._reply(h, 'error', 'expired')
                if not self._insert(h, binding, 'active'):
                    return self._reply(h, 'error', 'metadata_full')
                now = time.monotonic()
                entry = dict(job=None, lease=now + LEASE_SECONDS,
                             deadline=now + max(0, deadline - time.time()))
                self.jobs[h['job_id']] = entry
                # Durable binding precedes launch; stop and start share this lock.
                if self._row(h['job_id'])['state'] != 'active':
                    return self._reply(h, 'error', 'stopped')
                try:
                    entry['job'] = self.backend.start(payload, h['mime_type'], h['options'],
                                                     deadline=deadline,
                                                     lease_seconds=LEASE_SECONDS)
                except Exception:
                    self._stop(h['job_id'])
                    return self._reply(h, 'error', 'job_failed')
                return self._status(h, entry)
            if row is None:
                return self._reply(h, 'error', 'unknown_job')
            if row['state'] == 'stopping':
                return self._reply(h, 'error', 'stop_pending')
            if h['renew_seq'] <= row['seq']:
                return self._reply(h, 'error', 'renewal_replay')
            with self.db:
                self.db.execute('UPDATE jobs SET seq=? WHERE job_id=?', (h['renew_seq'], h['job_id']))
            entry = self.jobs[h['job_id']]
            try:
                renewed = entry['job'].renew(h['renew_seq'])
            except Exception:
                self._stop(h['job_id'])
                return self._reply(h, 'error', 'renewal_failed')
            if renewed:
                entry['lease'] = min(time.monotonic() + LEASE_SECONDS, entry['deadline'])
            return self._status(h, entry)


# Fixed deployment values; test constructors may supply temporary Unix paths.
SOCKET_PATH = '/run/gmail-attachment-rpc/manager.sock'
STATE_PATH = '/var/lib/gmail-attachment-rpc'
CONTROLLER_ACCOUNT = 'gmail-attachment-rpc'
MAX_CONNECTIONS = 4
READ_SECONDS = 5


class Server:
    def __init__(self, manager, socket_path, *, controller_uid):
        import socket
        self.manager = manager
        self.controller_uid = controller_uid
        self.socket_path = Path(socket_path)
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self.threads = set()
        self.threads_lock = threading.Lock()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.listener.bind(str(socket_path))
            os.chmod(socket_path, 0o660)
            self.socket_inode = self.socket_path.lstat().st_ino
            self.listener.listen(MAX_CONNECTIONS)
            self.listener.settimeout(.1)
        except BaseException:
            self.listener.close()
            raise

    def _connection(self, peer):
        import socket
        import struct
        try:
            with peer:
                _, uid, _ = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != self.controller_uid:
                    return
                end = time.monotonic() + READ_SECONDS
                def read(size):
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('RPC input deadline')
                    peer.settimeout(remaining)
                    return peer.recv(size)
                request, payload = rpc.read_frame(read)
                if read(1) != b'':
                    raise rpc.ProtocolError('Trailing RPC input')
                response, output = self.manager.handle(request, payload, peer_uid=uid)
                del payload
                peer.settimeout(READ_SECONDS)
                peer.sendall(rpc.encode_frame(response, output, response=True))
        except Exception:
            # Invalid/incomplete requests get EOF, never a success acknowledgement.
            pass
        finally:
            with self.threads_lock:
                self.threads.discard(threading.current_thread())
            self.slots.release()

    def serve(self, stop_event):
        import socket
        while not stop_event.is_set():
            self.manager.tick()
            try:
                peer, _ = self.listener.accept()
            except socket.timeout:
                continue
            if not self.slots.acquire(blocking=False):
                peer.close()
                continue
            thread = threading.Thread(target=self._connection, args=(peer,), daemon=False)
            with self.threads_lock:
                self.threads.add(thread)
            try:
                thread.start()
            except BaseException:
                peer.close()
                with self.threads_lock:
                    self.threads.discard(thread)
                self.slots.release()
                raise

    def close(self):
        self.listener.close()
        with self.threads_lock:
            threads = tuple(self.threads)
        for thread in threads:
            thread.join()
        try:
            if self.socket_path.lstat().st_ino == self.socket_inode:
                self.socket_path.unlink()
        except FileNotFoundError:
            pass


def main():
    import pwd
    import signal
    import sys
    from attachment_backend import AttachmentFirecrackerBackend, boundary
    if len(sys.argv) != 1 or os.geteuid() != 0:
        raise RuntimeError('Root fixed-command manager required')
    boundary()
    account = pwd.getpwnam(CONTROLLER_ACCOUNT)
    runtime = Path(SOCKET_PATH).parent
    runtime.mkdir(mode=0o750, exist_ok=True)
    info = runtime.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o750:
        raise RuntimeError('Unsafe manager runtime directory')
    os.chown(runtime, 0, account.pw_gid)
    manager = Manager(STATE_PATH, AttachmentFirecrackerBackend(), controller_uid=account.pw_uid)
    server = None
    try:
        path = Path(SOCKET_PATH)
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != 0:
                raise RuntimeError('Unsafe manager socket')
            path.unlink()  # Exclusive manager flock was acquired before this.
        server = Server(manager, SOCKET_PATH, controller_uid=account.pw_uid)
        os.chown(SOCKET_PATH, 0, account.pw_gid)
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        server.serve(stop)
    finally:
        if server is not None:
            server.close()
        try:
            manager.backend.reconcile()
        finally:
            manager.close()


if __name__ == '__main__':
    main()
