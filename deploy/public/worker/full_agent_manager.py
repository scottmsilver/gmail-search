#!/usr/bin/env python3
"""Fixed synthetic full-agent manager. Persistent bindings; transient bootstrap."""
from dataclasses import dataclass
import fcntl
import hashlib
import io
import os
from pathlib import Path
import socket
import sqlite3
import stat
import sys
import threading
import time
from types import SimpleNamespace

if __name__=='__main__':sys.path.insert(0,'/opt/gmail-worker')
import full_agent_rpc as rpc
from guest_agent_bootstrap import read_config

SOCKET_PATH='/run/gmail-full-agent-rpc/manager.sock'
STATE_PATH='/var/lib/gmail-full-agent-rpc'
ACCOUNT='gmail-full-agent-rpc'
LEASE_SECONDS=20
MAX_RECORDS=10000


@dataclass(frozen=True)
class Limits:
    vcpus:int=1
    memory_mib:int=1024
    pids:int=128
    disk_bytes:int=1024**3
    output_bytes:int=8*1024**2
    wall_seconds:int=180


def private(path,mode,directory=False):
    info=path.lstat()
    if (info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)!=mode or
        not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)) or
        (not directory and info.st_nlink!=1)):raise RuntimeError('Unsafe manager state')


class Manager:
    def __init__(self,state_dir,backend,*,controller_uid,session_factory):
        self.backend,self.controller_uid,self.session_factory=backend,controller_uid,session_factory
        self.lock=threading.RLock();self.jobs={};self.db=None;self.fd=None;self.closed=False
        root=Path(state_dir);root.mkdir(mode=0o700,exist_ok=True);private(root,0o700,True)
        try:
            self.fd=os.open(root/'manager.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
            private(root/'manager.lock',0o600);fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            path=root/'jobs.sqlite';fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC,0o600);os.close(fd);private(path,0o600)
            self.db=sqlite3.connect(path,check_same_thread=False);self.db.row_factory=sqlite3.Row
            self.db.execute('PRAGMA journal_mode=DELETE');self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS jobs(handle TEXT PRIMARY KEY, controller INTEGER NOT NULL, context TEXT NOT NULL, binding TEXT, state TEXT NOT NULL, seq INTEGER NOT NULL DEFAULT 0)');self.db.commit()
            fd=os.open(root,os.O_DIRECTORY|os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            # The shared synthetic physical namespace must be dedicated to this
            # manager: restart reaps all old VMs before any new request is served.
            for handle in backend.inventory():backend.stop(handle)
            if backend.inventory():raise RuntimeError('Unreconciled worker inventory')
            with self.db:self.db.execute("UPDATE jobs SET state='stopped'")
        except BaseException:
            if self.db is not None:self.db.close();self.db=None
            if self.fd is not None:os.close(self.fd);self.fd=None
            raise

    def _row(self,h):return self.db.execute('SELECT * FROM jobs WHERE handle=?',(h,)).fetchone()

    def _state(self,h,state):
        with self.lock,self.db:self.db.execute('UPDATE jobs SET state=? WHERE handle=?',(state,h))

    def _cleanup(self,h,entry):
        with entry['cleanup']:
            failed=False
            session=entry['session']
            if session is not None:
                try:session.close()
                except Exception:failed=True
            try:
                self.backend.stop(h)
                if h in self.backend.inventory():failed=True
            except Exception:failed=True
            self._state(h,'stopping' if failed else 'stopped')
            return not failed

    def _run(self,h,context,payload,entry):
        try:
            if entry['cancel'].is_set():return
            first_expiry=entry['expires']
            self.backend.launch(h,SimpleNamespace(**context,lease_expires=first_expiry),Limits())
            with self.lock:
                entry['launched']=True
                if entry['cancel'].is_set():return
                if entry['expires']>first_expiry:
                    self.backend.renew(h,SimpleNamespace(**context,lease_expires=entry['expires']))
            entry['session']=self.session_factory(h)
            entry['session'].bootstrap(h,payload,entry['cancel'])
            del payload
            while not entry['cancel'].wait(.1):
                if time.time()>=min(entry['expires'],context['deadline']):break
        except Exception:
            pass  # Never retain exceptions/tracebacks carrying capabilities.
        finally:
            self._state(h,'stopping');self._cleanup(h,entry)

    def _stop(self,h):
        with self.lock:
            row=self._row(h)
            if row['state']=='stopped' and h not in self.backend.inventory():
                old=self.jobs.get(h)
                if old is None or not old['thread'].is_alive():
                    self.jobs.pop(h,None);return True
            self._state(h,'stopping');entry=self.jobs.get(h)
            if entry is not None:entry['cancel'].set()
        if entry is not None:
            entry['thread'].join(timeout=20)
            if entry['thread'].is_alive():return False
            if not self._cleanup(h,entry):return False
        else:
            try:
                self.backend.stop(h)
                if h in self.backend.inventory():return False
            except Exception:return False
            self._state(h,'stopped')
        with self.lock:self.jobs.pop(h,None)
        return True

    def handle(self,h,payload,*,peer_uid):
        rpc.encode_frame(h,payload)
        if type(peer_uid) is not int or peer_uid!=self.controller_uid:return rpc.reply(h,'error','denied')
        op=h['op'];handle=h['handle']
        if op=='launch':
            try:read_config(io.BytesIO(payload).read)
            except ValueError:return rpc.reply(h,'error','invalid_bootstrap')
        with self.lock:
            if self.closed:return rpc.reply(h,'error','closed')
            if op=='inventory':
                owned={r[0] for r in self.db.execute("SELECT handle FROM jobs WHERE state!='stopped'")}
                return rpc.reply(h,'ok',handles=owned|self.backend.inventory())
            row=self._row(handle)
            if row is not None and (row['controller']!=peer_uid or (row['context']!=h['context_sha256'] and not (op=='stop' and h['context_sha256']=='0'*64))):return rpc.reply(h,'error','binding_mismatch')
            if row is None and op in ('stop','launch'):
                if self.db.execute('SELECT count(*) FROM jobs').fetchone()[0]>=MAX_RECORDS:return rpc.reply(h,'error','metadata_full')
                if op=='stop':
                    with self.db:self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,0)',(handle,peer_uid,h['context_sha256'],None,'stopping'))
                    row=self._row(handle)
            if row is not None and row['state']=='stopped' and op!='stop':return rpc.reply(h,'error','stopped')
            if op=='launch':
                binding=hashlib.sha256(rpc.canonical({'context':h['context'],'input_sha256':h['input_sha256']})).hexdigest()
                if row is not None:
                    if row['binding']!=binding:return rpc.reply(h,'error','binding_mismatch')
                    return rpc.reply(h,'running' if row['state']=='active' else 'error','ok' if row['state']=='active' else 'stop_pending')
                if self.db.execute("SELECT 1 FROM jobs WHERE state!='stopped'").fetchone():return rpc.reply(h,'error','busy')
                now=time.time();context=dict(h['context']);expires=min(h['lease_expires'],now+LEASE_SECONDS,context['deadline'])
                if not now<expires or context['deadline']-now>180:return rpc.reply(h,'error','expired')
                with self.db:self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,0)',(handle,peer_uid,h['context_sha256'],binding,'active'))
                entry=dict(cancel=threading.Event(),cleanup=threading.Lock(),session=None,launched=False,expires=expires,context=context)
                thread=threading.Thread(target=self._run,args=(handle,context,payload,entry),daemon=False);entry['thread']=thread;self.jobs[handle]=entry
                try:thread.start()
                except BaseException:
                    self.jobs.pop(handle,None);self._state(handle,'stopped');raise
                return rpc.reply(h,'running')
            if row is None:return rpc.reply(h,'error','unknown_job')
            if op=='renew':
                entry=self.jobs.get(handle)
                if row['state']!='active' or entry is None:return rpc.reply(h,'error','stop_pending')
                if h['renew_seq']<=row['seq']:return rpc.reply(h,'error','renewal_replay')
                now=time.time();expires=min(h['lease_expires'],now+LEASE_SECONDS,entry['context']['deadline'])
                if now>=entry['expires'] or now>=expires or h['lease_expires']>entry['context']['deadline']:return rpc.reply(h,'error','expired')
                with self.db:self.db.execute('UPDATE jobs SET seq=? WHERE handle=?',(h['renew_seq'],handle))
                if entry['launched']:
                    try:self.backend.renew(handle,SimpleNamespace(**entry['context'],lease_expires=expires))
                    except Exception:
                        entry['cancel'].set();self._state(handle,'stopping');return rpc.reply(h,'error','renewal_failed')
                entry['expires']=expires
                return rpc.reply(h,'running')
        # Do not hold the metadata lock while waiting for lifecycle cleanup.
        return rpc.reply(h,'stopped') if self._stop(handle) else rpc.reply(h,'error','stop_pending')

    def tick(self):
        with self.lock:
            for entry in self.jobs.values():
                if time.time()>=entry['expires']:entry['cancel'].set()

    def close(self):
        with self.lock:
            if self.db is None:return
            self.closed=True
            handles=[r[0] for r in self.db.execute("SELECT handle FROM jobs WHERE state!='stopped'")]
        failed=False
        for h in handles:
            if not self._stop(h):failed=True
        if failed:raise RuntimeError('Worker teardown pending')
        if self.db is not None:self.db.close();self.db=None
        if self.fd is not None:os.close(self.fd);self.fd=None


class RuntimeSession:
    """In-process relay sockets die with manager; close drains request threads."""
    def __init__(self,handle):
        from firecracker_backend import JAILS
        self.root=JAILS/handle/'root';self.relay=None;self.thread=None;self.listener=None

    def bootstrap(self,handle,payload,cancel):
        from vsock_http_relay import Relay
        class OwnedRelay(Relay):
            daemon_threads=False
            def __init__(self,*args):
                self.active=set();self.active_lock=threading.Lock();super().__init__(*args)
            def process_request(self,request,address):
                import socketserver
                if not self.slots.acquire(blocking=False):
                    request.close();return
                with self.active_lock:self.active.add(request)
                try:socketserver.ThreadingMixIn.process_request(self,request,address)
                except BaseException:
                    with self.active_lock:self.active.discard(request)
                    self.slots.release();request.close();raise
            def process_request_thread(self,request,address):
                try:super().process_request_thread(request,address)
                finally:
                    with self.active_lock:self.active.discard(request)
            def drain(self):
                with self.active_lock:
                    for peer in self.active:
                        try:peer.shutdown(socket.SHUT_RDWR)
                        except OSError:pass
                self.server_close()
        path=self.root/'gateway.vsock_8002'
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            self.listener=listener;listener.bind(str(path));os.chown(path,65534,65534);os.chmod(path,0o600);listener.listen(1);listener.settimeout(.1)
            self.relay=OwnedRelay(str(self.root/'gateway.vsock_8000'),18081);os.chown(self.root/'gateway.vsock_8000',65534,65534)
            self.thread=threading.Thread(target=self.relay.serve_forever,kwargs={'poll_interval':.1},daemon=False);self.thread.start()
            end=time.monotonic()+10
            while not cancel.is_set():
                if time.monotonic()>=end:raise TimeoutError('Bootstrap unavailable')
                try:peer,_=listener.accept()
                except socket.timeout:continue
                with peer:
                    # Only the jailed VMM UID can connect to its private path.
                    import struct
                    _,uid,_=struct.unpack('3i',peer.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                    if uid!=65534:raise RuntimeError('Unexpected bootstrap peer')
                    if cancel.is_set():return
                    peer.settimeout(2);peer.sendall(payload);peer.shutdown(socket.SHUT_WR)
                return
        self.listener=None

    def close(self):
        if self.relay is not None:
            if self.thread is not None and self.thread.is_alive():self.relay.shutdown()
            self.relay.drain()
            if self.thread is not None and self.thread.ident is not None:self.thread.join(timeout=5)
            if self.thread is not None and self.thread.is_alive():raise RuntimeError('Relay teardown pending')
            self.relay=None


def main(backend_factory=None):
    import pwd
    import signal
    from firecracker_backend import SyntheticFullAgentBackend,boundary
    from full_agent_rpc_server import Server
    if len(sys.argv)!=1 or os.geteuid()!=0:raise RuntimeError('Fixed root manager required')
    if backend_factory is None:
        boundary();backend_factory=SyntheticFullAgentBackend
    backend=backend_factory()
    account=pwd.getpwnam(ACCOUNT);root=Path(SOCKET_PATH).parent
    root.mkdir(mode=0o750,exist_ok=True);private(root,0o750,True);os.chown(root,0,account.pw_gid)
    manager=Manager(STATE_PATH,backend,controller_uid=account.pw_uid,session_factory=RuntimeSession)
    server=None
    try:
        path=Path(SOCKET_PATH)
        if path.exists() or path.is_symlink():
            info=path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid!=0:raise RuntimeError('Unsafe manager socket')
            path.unlink()
        server=Server(manager,path,controller_uid=account.pw_uid);os.chown(path,0,account.pw_gid)
        stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set());server.serve(stop)
    finally:
        if server is not None:server.close()
        manager.close()


if __name__=='__main__':main()
