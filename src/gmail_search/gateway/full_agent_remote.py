"""Trusted WorkerBackend adapter for the fixed synthetic full-agent manager."""
import hashlib
import io
import json
import threading
import uuid

from . import full_agent_rpc as rpc
from .attachment_remote import SSHTransport as _SSHTransport,exchange_process
from .registry import AccessDenied,RunLease
from .worker import WorkerLimits

FULL_LIMITS=WorkerLimits(vcpus=1,memory_mib=1024,pids=128,disk_bytes=1024**3,output_bytes=8*1024**2,wall_seconds=180)
# The runtime a browser may choose, and the guest profile each one boots.
GUEST_PROFILES={'pi':'mail-agent-pi-v1','pi_gemini':'mail-agent-pi-gemini-v1','claude':'mail-agent-claude-v1'}


def context(lease):
    result={key:getattr(lease,key) for key in rpc.CONTEXT};rpc.context_digest(result);return result


def validate_bootstrap(packet,prompt,runtime='pi'):
    if runtime not in GUEST_PROFILES:raise AccessDenied()
    if type(packet) is not bytes or not 4<len(packet)<=rpc.MAX_INPUT or int.from_bytes(packet[:4],'big')!=len(packet)-4:raise AccessDenied()
    try:
        value=json.loads(packet[4:],object_pairs_hook=rpc.pairs)
        if (type(value) is not dict or set(value)!={'version','profile','prompt','tool_config','inference_capability','events_capability'} or
            type(value['version']) is not int or value['version']!=1 or value['profile']!=GUEST_PROFILES[runtime] or value['prompt']!=prompt):raise ValueError()
        if type(prompt) is not str or not prompt.strip() or '\x00' in prompt or len(prompt.encode())>16384:raise ValueError()
        cfg=value['tool_config']
        if type(cfg) is not dict or set(cfg)!={'version','tool_profile','capabilities'} or type(cfg['version']) is not int or cfg['version']!=3 or cfg['tool_profile']!='mail-raw-mcp-v3':raise ValueError()
        caps=cfg['capabilities']
        if type(caps) is not dict or set(caps)!={'sql','retrieval','artifact','attachment'}:raise ValueError()
        for token in [*caps.values(),value['inference_capability'],value['events_capability']]:rpc.hex_value(token,64)
    except (ValueError,TypeError,UnicodeError,RecursionError):raise AccessDenied() from None


class SSHTransport(_SSHTransport):
    def command(self):
        command=super().command();command[-2:]=['gmail-full-agent-rpc@'+self.host,'full-agent-rpc-v1'];return command

    def exchange(self,header,payload,cancelled):
        wire=exchange_process(self.command(),rpc.encode_frame(header,payload),cancelled,timeout=30,max_output=rpc.MAX_OUTPUT)
        stream=io.BytesIO(wire);response,body=rpc.read_frame(stream.read,response=True)
        if stream.read(1):raise AccessDenied()
        return response,body


class SSHFullAgentBackend:
    namespace='synthetic-full-agent-rpc'

    def __init__(self,registry,*,transport,envelope_for):
        if not callable(envelope_for):raise ValueError('Trusted envelope factory required')
        self.registry,self.transport,self.envelope_for=registry,transport,envelope_for
        self._lock=threading.RLock();self._inputs={};self._closed=False
        with registry._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS full_agent_remote_jobs(handle TEXT PRIMARY KEY,run_id TEXT NOT NULL UNIQUE,context TEXT NOT NULL,digest TEXT NOT NULL,input_digest TEXT NOT NULL,state TEXT NOT NULL,seq INTEGER NOT NULL DEFAULT 0)')

    def _active(self,lease):
        if type(lease) is not RunLease:raise AccessDenied()
        with self.registry._transaction() as db:
            row=self.registry._active(db,lease.run_id);self.registry._fence(db,row);current=self.registry._lease(row)
        if any(getattr(current,k)!=getattr(lease,k) for k in RunLease.__dataclass_fields__ if k!='lease_expires'):raise AccessDenied()
        return current

    def _prune(self):
        for run_id,entry in tuple(self._inputs.items()):
            try:self._active(entry['lease'])
            except AccessDenied:self._inputs.pop(run_id,None)

    def prepare_input(self,lease,prompt,runtime='pi'):
        self._active(lease)
        payload=self.envelope_for(lease,prompt,runtime);validate_bootstrap(payload,prompt,runtime)
        self._active(lease)
        with self._lock:
            if self._closed:raise AccessDenied()
            self._prune();old=self._inputs.get(lease.run_id)
            if old is not None:
                if old['payload']!=payload or context(old['lease'])!=context(lease):raise AccessDenied()
                return
            if len(self._inputs)>=8:raise AccessDenied()
            self._inputs[lease.run_id]={'lease':lease,'payload':payload}

    def _call(self,op,handle=rpc.ZERO,digest='0'*64,payload=b'',**extra):
        header=dict(version=1,op=op,request_id=uuid.uuid4().hex,handle=handle,context_sha256=digest,payload_size=len(payload),**extra)
        rpc.encode_frame(header,payload)
        try:
            response,body=self.transport.exchange(header,payload,threading.Event());rpc.encode_frame(response,body,response=True)
            if body or any(response[k]!=header[k] for k in rpc.COMMON-{'payload_size'}) or response['status']=='error':raise AccessDenied()
            return response
        except Exception:raise AccessDenied() from None

    def launch(self,handle,lease,limits):
        rpc.hex_value(handle,32)
        if limits!=FULL_LIMITS or type(limits) is not WorkerLimits:raise AccessDenied()
        with self._lock:
            if self._closed:raise AccessDenied()
            current=self._active(lease);ctx=context(current);digest=rpc.context_digest(ctx)
            with self.registry._transaction() as db:old=db.execute('SELECT * FROM full_agent_remote_jobs WHERE handle=?',(handle,)).fetchone()
            if old is not None and (old['run_id']!=lease.run_id or old['digest']!=digest or old['state']=='stopped'):raise AccessDenied()
            if old is not None and old['state']=='active':
                if handle not in self.inventory():raise AccessDenied()
                return
            entry=self._inputs.get(lease.run_id)
            if entry is None or context(entry['lease'])!=ctx:raise AccessDenied()
            payload=entry['payload'];input_digest=hashlib.sha256(payload).hexdigest()
            if old is not None and old['input_digest']!=input_digest:raise AccessDenied()
            if old is None:
                with self.registry._transaction() as db:
                    if db.execute('SELECT count(*) FROM full_agent_remote_jobs').fetchone()[0]>=10000:raise AccessDenied()
                    db.execute('INSERT INTO full_agent_remote_jobs VALUES(?,?,?,?,?,?,0)',(handle,lease.run_id,json.dumps(ctx),digest,input_digest,'pending'))
            self._active(lease)
            response=self._call('launch',handle,digest,payload,context=ctx,input_sha256=input_digest,lease_expires=current.lease_expires)
            if response['status']!='running':raise AccessDenied()
            self._active(lease)
            with self.registry._transaction() as db:db.execute("UPDATE full_agent_remote_jobs SET state='active' WHERE handle=?",(handle,))
            self._inputs.pop(lease.run_id,None)

    def renew(self,handle,lease):
        with self._lock:
            if self._closed:raise AccessDenied()
            current=self._active(lease)
            with self.registry._transaction() as db:
                row=db.execute('SELECT * FROM full_agent_remote_jobs WHERE handle=?',(handle,)).fetchone()
                if row is None or row['state']!='active' or row['digest']!=rpc.context_digest(context(current)):raise AccessDenied()
                seq=row['seq']+1;db.execute('UPDATE full_agent_remote_jobs SET seq=? WHERE handle=?',(seq,handle))
            response=self._call('renew',handle,row['digest'],renew_seq=seq,lease_expires=current.lease_expires)
            if response['status']!='running':raise AccessDenied()
            self._active(lease)

    def stop(self,handle):
        rpc.hex_value(handle,32)
        with self._lock:
            with self.registry._transaction() as db:
                row=db.execute('SELECT * FROM full_agent_remote_jobs WHERE handle=?',(handle,)).fetchone()
                if row is not None:db.execute("UPDATE full_agent_remote_jobs SET state='stopping' WHERE handle=?",(handle,))
            response=self._call('stop',handle,row['digest'] if row else '0'*64)
            if response['status']!='stopped':raise AccessDenied()
            if row is not None:
                with self.registry._transaction() as db:db.execute("UPDATE full_agent_remote_jobs SET state='stopped' WHERE handle=?",(handle,))
                self._inputs.pop(row['run_id'],None)

    def inventory(self):
        with self._lock:
            response=self._call('inventory')
            if response['status']!='ok':raise AccessDenied()
            return set(response['handles'])

    def close(self):
        with self._lock:
            self._closed=True;self._inputs.clear()
            with self.registry._transaction() as db:rows=db.execute("SELECT handle FROM full_agent_remote_jobs WHERE state!='stopped'").fetchall()
            for row in rows:self.stop(row['handle'])
