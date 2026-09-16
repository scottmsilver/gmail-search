#!/usr/bin/env python3
"""One fixed parser VM per job, synthetic outer worker only; no guest paths.

Independent Firecracker supervisor enforces the 45s hard deadline and kills and
reaps the full VMM cgroup even if this controller dies. One job per outer worker
prevents callers from multiplying its memory/disk allowance. No inference relay.
"""
from dataclasses import dataclass
import fcntl
import json
import math
import os
import socket
import stat
import struct
import threading
import time
from types import SimpleNamespace
import uuid

from firecracker_backend import SyntheticAttachmentBackend, ROOT, JAILS, boundary


@dataclass
class Limits:
    vcpus: int = 1
    memory_mib: int = 768
    pids: int = 128
    disk_bytes: int = 1024**3
    output_bytes: int = 128*1024
    wall_seconds: int = 45


class AttachmentJob:
    """Cancellable host-side handle; the guest never receives this object.

    The worker owns launch and transport only. ``stop`` owns teardown and the
    capacity lock, including after transport completion or failure. Failed stop
    acknowledgements leave this handle in the controller's pending registry so
    a trusted caller can retry ``stop``; capacity is never released on failure.
    """

    def __init__(self, controller, backend, lock, data, header, deadline, lease_seconds):
        self.controller, self.backend, self.lock = controller, backend, lock
        self.handle = uuid.uuid4().hex
        self.run_id = uuid.uuid4().hex
        self.deadline, self.lease_seconds = deadline, lease_seconds
        self._launched = threading.Event()
        self._lease_lock = threading.Lock()
        self._renew_seq = 0
        self.cancelled = threading.Event()
        self._sockets_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._sockets = []
        self._stopped = False
        self.result = self.error = None
        self.thread = threading.Thread(target=self._work, args=(data, header),
                                       name='attachment-parser-job', daemon=False)

    @staticmethod
    def _close(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def register_socket(self, sock):
        with self._sockets_lock:
            if self.cancelled.is_set():
                self._close(sock)
                raise RuntimeError('Attachment job cancelled')
            self._sockets.append(sock)

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise RuntimeError('Attachment job cancelled')

    def _work(self, data, header):
        try:
            self.check_cancelled()
            self.result = self.controller._run(self.backend, data, header, job=self)
        except BaseException as exc:
            self.error = exc

    def wait(self):
        self.thread.join()
        if self.error is not None:
            raise self.error
        self.check_cancelled()
        return self.result

    def renew(self, renew_seq):
        """Only trusted, freshly authorized manager polls may call this."""
        with self._lease_lock:
            if type(renew_seq) is not int or not self._renew_seq < renew_seq <= 9223372036854775807:
                raise ValueError('Invalid lease renewal sequence')
            self.check_cancelled()
            if time.time() >= self.deadline:
                raise RuntimeError('Parser deadline expired')
            if not self._launched.is_set():
                return False
            lease = SimpleNamespace(run_id=self.run_id, deadline=self.deadline,
                lease_expires=min(self.deadline, time.time()+self.lease_seconds))
            self.backend.renew(self.handle, lease)
            self._renew_seq = renew_seq
            return True

    def stop(self):
        self.cancelled.set()
        with self._sockets_lock:
            for sock in self._sockets:
                self._close(sock)
        with self._stop_lock:
            if self._stopped:
                return
            # FirecrackerBackend records a stop tombstone even before launch.
            # A racing worker cannot create a VM after this acknowledgement.
            try:
                self.backend.stop(self.handle)
            finally:
                self.thread.join(timeout=20)
            if self.thread.is_alive():
                raise RuntimeError('Parser controller has not stopped; retain binding and retry')
            os.close(self.lock)
            self._stopped = True
            with self.controller._jobs_lock:
                self.controller._jobs.pop(self.handle, None)


class AttachmentFirecrackerBackend:
    def __init__(self):
        self._jobs = {}
        self._jobs_lock = threading.Lock()

    def pending_jobs(self):
        """Trusted reconciliation can retry failed teardown through these handles."""
        with self._jobs_lock:
            return tuple(self._jobs.values())

    def start(self, data, mime_type, options, *, deadline=None, lease_seconds=3):
        """Reserve capacity and return a handle; launch/I/O run off the caller loop."""
        boundary()
        now = time.time()
        if deadline is None:
            deadline = now + 45
        if (type(deadline) not in (int, float) or not math.isfinite(deadline) or not now < deadline <= now + 45
                or type(lease_seconds) not in (int, float) or not math.isfinite(lease_seconds)
                or not 0 < lease_seconds <= 45):
            raise ValueError('Invalid parser lease')
        if type(data) is not bytes or not 0<len(data)<=10*1024**2:
            raise ValueError('Input limit')
        header = json.dumps(dict(size=len(data),mime_type=mime_type,options=options)).encode()
        if len(header)>1024:
            raise ValueError('Header limit')
        backend = SyntheticAttachmentBackend()
        lock = os.open(ROOT/'attachment.lock', os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC, 0o600)
        try:
            info = os.fstat(lock)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
                raise RuntimeError('Unsafe parser capacity lock')
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if backend.inventory():
                raise RuntimeError('Existing worker jobs require reconciliation before parser admission')
            job = AttachmentJob(self, backend, lock, data, header, deadline, lease_seconds)
            with self._jobs_lock:
                self._jobs[job.handle] = job
            try:
                job.thread.start()
            except BaseException:
                with self._jobs_lock:
                    self._jobs.pop(job.handle, None)
                raise
            return job
        except BaseException:
            os.close(lock)
            raise

    def parse(self, data, mime_type, options):
        """Compatibility API; even successful calls await complete VM teardown."""
        job = self.start(data, mime_type, options, lease_seconds=45)
        try:
            return job.wait()
        finally:
            job.stop()

    def reconcile(self):
        """Trusted startup recovery; refuse admission until old work is reaped."""
        boundary()
        for job in self.pending_jobs():
            job.stop()
        lock = os.open(ROOT/'attachment.lock', os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC, 0o600)
        try:
            info = os.fstat(lock)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
                raise RuntimeError('Unsafe parser capacity lock')
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            backend = SyntheticAttachmentBackend()
            for handle in tuple(backend.inventory()):
                backend.stop(handle)
            if backend.inventory():
                raise RuntimeError('Worker reconciliation incomplete')
        finally:
            os.close(lock)

    def _run(self, backend, data, header, *, job=None):
        handle = job.handle if job is not None else uuid.uuid4().hex
        deadline = job.deadline if job is not None else time.time()+45
        lease = SimpleNamespace(run_id=job.run_id if job is not None else uuid.uuid4().hex,
            lease_expires=min(deadline, time.time()+job.lease_seconds) if job is not None else deadline,
            deadline=deadline)
        server = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            if job is not None:
                job.register_socket(server)
                job.check_cancelled()
            backend.launch(handle,lease,Limits())
            if job is not None:
                job._launched.set()
                job.check_cancelled()
            path = JAILS/handle/'root/gateway.vsock_8002'
            server.bind(str(path))
            os.chown(path,65534,65534)
            os.chmod(path,0o600)
            server.listen(1)
            server.settimeout(20)
            peer,_ = server.accept()
            if job is not None:
                job.register_socket(peer)
            end = time.monotonic()+30
            with peer:
                peer.settimeout(15)
                peer.sendall(struct.pack('!I',len(header))+header+data)
                def exact(size):
                    output = bytearray()
                    while len(output)<size:
                        remaining=end-time.monotonic()
                        if remaining<=0:
                            raise TimeoutError()
                        peer.settimeout(remaining)
                        part=peer.recv(min(65536,size-len(output)))
                        if not part:
                            raise ValueError('Guest disconnected')
                        output.extend(part)
                    return bytes(output)
                size = struct.unpack('!I',exact(4))[0]
                if size>8*1024**2:
                    raise ValueError('Output limit')
                return exact(size)
        finally:
            server.close()
            if job is None:
                backend.stop(handle)
