"""Pinned SSH transport to the fixed attachment RPC frontend.

No shell command construction, inherited SSH configuration, forwarding, or
unbounded subprocess output. Platform keys and capability secrets stay here.
"""
import io
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import threading
import time
import uuid

from .attachment_rpc import MAX_OUTPUT, context_digest, encode_frame, read_frame


class RemoteAttachmentUnavailable(RuntimeError):
    def __init__(self):
        super().__init__('Isolated attachment worker unavailable')


def exchange_process(command, packet, cancelled, *, timeout=10, max_output=MAX_OUTPUT):
    """Trusted argv only; bounded bidirectional I/O and complete child reaping."""
    if cancelled.is_set():
        raise RemoteAttachmentUnavailable()
    proc = None
    selector = selectors.DefaultSelector()
    output = bytearray()
    deadline = time.monotonic()+timeout
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, close_fds=True, start_new_session=True,
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'})
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            os.set_blocking(stream.fileno(), False)
        selector.register(proc.stdout, selectors.EVENT_READ, 'stdout')
        selector.register(proc.stderr, selectors.EVENT_READ, 'stderr')
        if packet:
            selector.register(proc.stdin, selectors.EVENT_WRITE, 'stdin')
        else:
            proc.stdin.close()
        offset, errors = 0, 0
        while selector.get_map():
            remaining = deadline-time.monotonic()
            if cancelled.is_set() or remaining <= 0:
                raise RemoteAttachmentUnavailable()
            for key, _ in selector.select(min(.05, remaining)):
                if key.data == 'stdin':
                    try:
                        offset += os.write(key.fd, memoryview(packet)[offset:offset+65536])
                    except BlockingIOError:
                        continue
                    if offset == len(packet):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()  # protocol requires EOF after one frame
                else:
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    elif key.data == 'stdout':
                        if len(output)+len(chunk) > max_output:
                            raise RemoteAttachmentUnavailable()
                        output.extend(chunk)
                    else:
                        errors += len(chunk)
                        if errors > 16384:
                            raise RemoteAttachmentUnavailable()
        if cancelled.is_set() or proc.wait(timeout=max(.001, deadline-time.monotonic())) != 0:
            raise RemoteAttachmentUnavailable()
        return bytes(output)
    except (OSError, subprocess.SubprocessError):
        raise RemoteAttachmentUnavailable() from None
    finally:
        selector.close()
        if proc is not None:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=2)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()


def decode_response(request, wire):
    try:
        stream = io.BytesIO(wire)
        response, payload = read_frame(stream.read, response=True)
        if stream.read(1) or any(response[key] != request[key] for key in
                               ('version', 'op', 'request_id', 'job_id', 'context_sha256')):
            raise RemoteAttachmentUnavailable()
        return response, payload
    except (ValueError, TypeError, KeyError):
        raise RemoteAttachmentUnavailable() from None


class SSHTransport:
    def __init__(self, *, host, port, private_key, known_hosts):
        if (type(host) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]{0,252}', host)
                or type(port) is not int or not 1 <= port <= 65535):
            raise ValueError('Invalid fixed worker endpoint')
        self.host, self.port = host, port
        self.private_key, self.known_hosts = Path(private_key), Path(known_hosts)
        for path, private in ((self.private_key, True), (self.known_hosts, False)):
            info = path.lstat()
            if (not path.is_absolute() or '..' in path.parts or not stat.S_ISREG(info.st_mode)
                    or info.st_uid not in (0, os.geteuid()) or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) & (0o077 if private else 0o022)):
                raise ValueError('Unsafe worker SSH identity configuration')

    def command(self):
        command = ['/usr/bin/ssh', '-F', '/dev/null', '-T', '-p', str(self.port), '-i', str(self.private_key)]
        for option in ('BatchMode=yes', 'IdentitiesOnly=yes', 'IdentityAgent=none', 'ForwardAgent=no',
                       'ClearAllForwardings=yes', 'StrictHostKeyChecking=yes', 'ProxyCommand=none',
                       'ProxyJump=none', 'GlobalKnownHostsFile=/dev/null',
                       'UserKnownHostsFile='+str(self.known_hosts), 'ConnectTimeout=3',
                       'ServerAliveInterval=1', 'ServerAliveCountMax=2', 'RequestTTY=no',
                       'PermitLocalCommand=no', 'ControlMaster=no', 'ControlPath=none', 'UpdateHostKeys=no'):
            command.extend(('-o', option))
        return command + ['gmail-attachment-rpc@'+self.host, 'attachment-rpc-v1']

    def exchange(self, header, payload, cancelled):
        wire = exchange_process(self.command(), encode_frame(header, payload), cancelled)
        return decode_response(header, wire)


class SSHAttachmentBackend:
    """Trusted controller adapter; unresolved jobs block startup until reconciled.

    Registry contains identifiers and hashes only. An authorization callback
    stays on the host and is called before every lease-renewing RPC.
    """
    def __init__(self, registry, *, transport, poll_interval=.5):
        if type(poll_interval) not in (int, float) or not 0 < poll_interval <= 1:
            raise ValueError('Invalid attachment poll interval')
        self.registry, self.transport, self.poll_interval = registry, transport, poll_interval
        self._jobs, self._lock = {}, threading.Lock()
        with registry._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS attachment_remote_jobs (
                job_id TEXT PRIMARY KEY, context TEXT NOT NULL, context_sha256 TEXT NOT NULL,
                input_sha256 TEXT NOT NULL, state TEXT NOT NULL)''')
            self._ready = not db.execute("SELECT 1 FROM attachment_remote_jobs WHERE state!='stopped' LIMIT 1").fetchone()

    def _state(self, job_id, state):
        with self.registry._transaction() as db:
            db.execute('UPDATE attachment_remote_jobs SET state=? WHERE job_id=?', (state, job_id))

    def _call(self, job_id, digest, op, *, payload=b'', cancelled=None, **extra):
        header = dict(version=1, op=op, request_id=str(uuid.uuid4()), job_id=job_id,
                      context_sha256=digest, payload_size=len(payload), **extra)
        # Validate before any transport. The concrete SSH transport validates
        # response framing, digest, EOF and all echoed request bindings.
        encode_frame(header, payload)
        response, body = self.transport.exchange(header, payload, cancelled or threading.Event())
        if (any(response[key] != header[key] for key in ('version', 'op', 'request_id', 'job_id', 'context_sha256'))
                or response['status'] == 'error'):
            raise RemoteAttachmentUnavailable()
        return response, body

    def start_authorized(self, data, mime_type, options, *, lease, attachment_id, authorize):
        context = dict(owner_id=lease.owner_id, run_id=lease.run_id, conversation_id=lease.conversation_id,
                       fence=lease.fence, attachment_id=attachment_id, deadline=min(lease.deadline, time.time()+45))
        digest = context_digest(context)
        input_digest = hashlib.sha256(data).hexdigest()
        job = _RemoteJob(self, context, digest, authorize)
        # Strictly validate input/options before persisting a binding or thread.
        encode_frame(dict(version=1, op='start', request_id=str(uuid.uuid4()), job_id=job.job_id,
                          context_sha256=digest, payload_size=len(data), context=context,
                          mime_type=mime_type, options=options, input_sha256=input_digest), data)
        with self._lock:
            if not self._ready:
                raise RemoteAttachmentUnavailable()
            with self.registry._transaction() as db:
                if db.execute('SELECT count(*) FROM attachment_remote_jobs').fetchone()[0] >= 10000:
                    raise RemoteAttachmentUnavailable()
                db.execute('INSERT INTO attachment_remote_jobs VALUES(?,?,?,?,?)',
                           (job.job_id, json.dumps(context), digest, input_digest, 'pending'))
            self._jobs[job.job_id] = job
            job.thread = threading.Thread(target=job._work, args=(data, mime_type, options, input_digest),
                                          name='attachment-ssh-job', daemon=False)
            try:
                job.thread.start()
            except BaseException:
                self._state(job.job_id, 'stopped')  # no transport was dispatched
                self._jobs.pop(job.job_id, None)
                raise
        return job

    def reconcile(self):
        """Cancel unresolved prior jobs; never replay input after a host restart."""
        with self._lock:
            self._ready = False
            jobs = tuple(self._jobs.values())
        for job in jobs:
            job.stop()
        with self.registry._transaction() as db:
            rows = db.execute("SELECT * FROM attachment_remote_jobs WHERE state!='stopped'").fetchall()
        for row in rows:
            if context_digest(json.loads(row['context'])) != row['context_sha256']:
                raise RemoteAttachmentUnavailable()
            response, _ = self._call(row['job_id'], row['context_sha256'], 'stop')
            if response['status'] != 'stopped':
                raise RemoteAttachmentUnavailable()
            self._state(row['job_id'], 'stopped')
        with self._lock:
            self._ready = True


class _RemoteJob:
    def __init__(self, backend, context, digest, authorize):
        self.backend, self.context, self.digest, self.authorize = backend, context, digest, authorize
        self.job_id = str(uuid.uuid4())
        self.cancelled, self._stop_lock = threading.Event(), threading.Lock()
        self.result = self.error = None
        self.thread = None
        self.stopped = False

    def _authorize(self):
        lease = self.authorize()
        if (self.cancelled.is_set() or time.time() >= self.context['deadline']
                or any(getattr(lease, name) != self.context[name] for name in
                       ('owner_id', 'run_id', 'conversation_id', 'fence'))
                or lease.deadline < self.context['deadline']):
            raise RemoteAttachmentUnavailable()

    def _work(self, data, mime_type, options, input_digest):
        try:
            self._authorize()
            response, body = self.backend._call(self.job_id, self.digest, 'start', payload=data,
                cancelled=self.cancelled, context=self.context, mime_type=mime_type,
                options=options, input_sha256=input_digest)
            sequence = 0
            while response['status'] == 'running':
                if self.cancelled.wait(self.backend.poll_interval):
                    raise RemoteAttachmentUnavailable()
                self._authorize()  # no autonomous renewal after capability loss
                sequence += 1
                response, body = self.backend._call(self.job_id, self.digest, 'poll',
                    cancelled=self.cancelled, renew_seq=sequence)
            if response['status'] != 'done':
                raise RemoteAttachmentUnavailable()
            self._authorize()
            self.result = body
        except BaseException as error:
            self.error = error if isinstance(error, PermissionError) else RemoteAttachmentUnavailable()

    def wait(self):
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.cancelled.is_set():
            raise RemoteAttachmentUnavailable()
        return self.result

    def stop(self):
        self.cancelled.set()
        with self._stop_lock:
            if self.stopped:
                return
            self.backend._state(self.job_id, 'stopping')
            try:
                response, _ = self.backend._call(self.job_id, self.digest, 'stop')
                if response['status'] != 'stopped':
                    raise RemoteAttachmentUnavailable()
            except Exception:
                raise RemoteAttachmentUnavailable() from None
            finally:
                self.thread.join(timeout=20)
            if self.thread.is_alive():
                raise RemoteAttachmentUnavailable()
            self.backend._state(self.job_id, 'stopped')
            self.stopped = True
            with self.backend._lock:
                self.backend._jobs.pop(self.job_id, None)
