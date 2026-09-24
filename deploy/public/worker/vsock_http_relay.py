#!/usr/bin/env python3
"""Synthetic worker relay: one guest socket, fixed loopback upstream, closed paths.

No CONNECT, arbitrary upstream, cookies, forwarded identity headers or HTTP
pipelining. Input framing is parsed and reconstructed before forwarding. This
is a synthetic qualification adapter, not a deployed authenticated worker link.
Firecracker guest-initiated protocol: https://github.com/firecracker-microvm/firecracker/blob/main/docs/vsock.md
"""
import argparse
import hashlib
import http.client
import http.server
import json
import os
from pathlib import Path
import re
import select
import socket
import socketserver
import stat
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

MAX_BODY = 20 * 1024**2
MAX_RESPONSE = 32 * 1024**2
MAX_RAW_RESPONSE = 4 + 4096 + 10 * 1024**2


class HeaderReader:
    """Bound header lines/bytes without changing process-global HTTP limits."""
    def __init__(self, stream):
        self.stream=stream
        self.total=0
        self.header_phase=True

    def readline(self, limit=-1):
        if not self.header_phase:
            return self.stream.readline(limit)
        line=self.stream.readline(min(limit,8193) if limit>=0 else 8193)
        self.total+=len(line)
        if len(line)>8192 or self.total>65536:
            raise http.client.LineTooLong('bounded relay header')
        return line

    def __getattr__(self, name):
        return getattr(self.stream,name)


class Relay(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, path, upstream_port, *, artifact_root=None):
        self.upstream_port = upstream_port
        self.artifact_root = artifact_root
        self.slots = threading.BoundedSemaphore(4)
        self.artifact_lock = threading.Lock()
        super().__init__(str(path), Handler)
        os.chmod(path, 0o600)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        pass  # Never log guest bytes, capabilities, or upstream diagnostics.


def _watch_guest(request, upstream_socket, stop):
    """A qualified guest keeps its write side open until the response ends.

    After its bounded request body, EOF or more bytes mean disconnect/protocol
    violation. Shutdown also interrupts a stalled getresponse/read in the relay.
    """
    while not stop.is_set():
        try:
            readable, _, _ = select.select([request], [], [], .05)
            if not readable:
                continue
            # No pipelining is supported; either EOF or extra input ends this
            # operation. Do not consume bytes or inspect them in diagnostics.
            upstream_socket.shutdown(socket.SHUT_RDWR)
        except (OSError, ValueError):
            pass
        return


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        self.request.settimeout(5)
        self.header_timer=threading.Timer(5,self._interrupt)
        self.header_timer.daemon=True
        self.header_timer.start()
        super().setup()
        self.rfile=HeaderReader(self.rfile)

    def _interrupt(self):
        try:
            self.request.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def parse_request(self):
        try:
            return super().parse_request()
        finally:
            self.header_timer.cancel()
            self.rfile.header_phase=False

    def finish(self):
        self.header_timer.cancel()
        super().finish()

    def log_message(self, *_):
        pass

    def do_CONNECT(self):
        self.send_error(405, 'Unsupported relay operation')

    def do_GET(self):
        self._relay()

    def do_POST(self):
        self._relay()

    def _route(self):
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.fragment or len(self.path)>4096:
            raise ValueError()
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        if self.command=='POST' and parsed.path in ('/v1/messages','/v1/messages/count_tokens'):
            if query not in ({},{'beta':['true']}):
                raise ValueError()
            return parsed.path
        if self.command=='POST' and self.path=='/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse':
            return self.path
        if self.command=='POST' and self.path=='/v1/chat/completions':  # OpenRouter-served Pi models
            return self.path
        if self.command=='POST' and parsed.path in ('/v1/sql', '/v1/thread', '/v1/search', '/v1/find-facts', '/v1/judge', '/v1/query-emails', '/v1/attachment/parse', '/v1/attachment/meta', '/v1/attachment/text', '/v1/attachment/list', '/v1/attachment/raw', '/v1/events') and not query:
            return parsed.path
        if self.command=='GET' and parsed.path=='/v1/schema' and not query:
            return parsed.path
        if self.command=='POST' and parsed.path=='/v1/artifacts':
            if set(query)!={'filename'} or len(query['filename'])!=1:
                raise ValueError()
            name=query['filename'][0]
            if not 1<=len(name.encode())<=200 or name in ('.','..') or any(ord(c)<32 or ord(c)==127 or c in '/\\' for c in name):
                raise ValueError()
            return self.path
        if self.command=='GET' and re.fullmatch('/v1/artifacts/[a-f0-9]{32}',parsed.path) and not query:
            return parsed.path
        raise ValueError()

    def _relay(self):
        upstream=None
        watcher=None
        watch_stop=threading.Event()
        started=False
        response_deadline=time.monotonic()+90
        self.close_connection=True
        try:
            route=self._route()
            raw_response = route == '/v1/attachment/raw'
            # One framing mechanism, one credential. No guest-selected routing
            # headers survive the reconstructed request.
            if self.headers.get_all('Transfer-Encoding') or self.headers.get_all('Cookie') or self.headers.get_all('Expect'):
                raise ValueError()
            lengths=self.headers.get_all('Content-Length') or []
            if len(lengths)>1 or (self.command=='POST' and len(lengths)!=1):
                raise ValueError()
            length=int(lengths[0]) if lengths and re.fullmatch('[0-9]{1,8}',lengths[0]) else 0
            if (lengths and not re.fullmatch('[0-9]{1,8}',lengths[0])) or not 0<=length<=MAX_BODY or (self.command=='GET' and length):
                raise ValueError()
            if raw_response and length > 4096:
                raise ValueError()
            auth=self.headers.get_all('Authorization') or []
            keys=(self.headers.get_all('x-api-key') or []) + (self.headers.get_all('x-goog-api-key') or [])
            if len(auth)+len(keys)!=1:
                raise ValueError()
            bearer=auth[0] if auth else 'Bearer '+keys[0]
            if not bearer.startswith('Bearer ') or not 8<=len(bearer)<=1024 or any(not 33<=ord(c)<=126 for c in bearer[7:]):
                raise ValueError()
            content_types=self.headers.get_all('Content-Type') or []
            if len(content_types)>1:
                raise ValueError()
            content_type=content_types[0].split(';',1)[0] if content_types else 'application/json'
            expected='application/octet-stream' if route.startswith('/v1/artifacts?') else 'application/json'
            if self.command=='POST' and content_type!=expected:
                raise ValueError()
            deadline=time.monotonic()+30
            chunks=[]
            left=length
            while left:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise ValueError()
                self.request.settimeout(remaining)
                chunk=self.rfile.read(min(left,65536))
                if not chunk:
                    raise ValueError()
                chunks.append(chunk)
                left-=len(chunk)
            body=b''.join(chunks)
            if route.startswith('/v1/artifacts?') and self.server.artifact_root is not None:
                self._artifact(body)
                return
            upstream=http.client.HTTPConnection('127.0.0.1',self.server.upstream_port,timeout=60)
            upstream.connect()
            watcher=threading.Thread(target=_watch_guest,args=(self.request,upstream.sock,watch_stop),daemon=True)
            watcher.start()
            upstream.request(self.command,route,body=body,headers={'Authorization':bearer,'Content-Type':content_type,'Connection':'close','Accept-Encoding':'identity'})
            response=upstream.getresponse()
            expected_status=201 if route.startswith('/v1/artifacts?') else 200
            if response.status!=expected_status:
                raise ValueError()
            response_length = None
            if raw_response:
                raw_types = response.headers.get_all('Content-Type') or []
                raw_lengths = response.headers.get_all('Content-Length') or []
                if (raw_types != ['application/octet-stream']
                        or len(raw_lengths) != 1
                        or not re.fullmatch('[1-9][0-9]{0,7}', raw_lengths[0])
                        or response.headers.get_all('Transfer-Encoding')
                        or response.headers.get_all('Content-Encoding')):
                    raise ValueError()
                response_length = int(raw_lengths[0])
                if not 4 < response_length <= MAX_RAW_RESPONSE:
                    raise ValueError()
            self.send_response(expected_status)
            self.send_header('Content-Type',response.getheader('Content-Type','application/octet-stream'))
            if response_length is not None:
                self.send_header('Content-Length', str(response_length))
            self.send_header('Connection','close')
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            started=True
            total=0
            while True:
                if response.isclosed():
                    break
                remaining=response_deadline-time.monotonic()
                if remaining<=0:
                    raise ValueError()
                response.fp.raw._sock.settimeout(min(60,remaining))
                chunk=response.read1(65536)
                if not chunk:
                    break
                total+=len(chunk)
                if total>(response_length if response_length is not None else MAX_RESPONSE):
                    raise ValueError()
                self.wfile.write(chunk)
                self.wfile.flush()
            if response_length is not None and total != response_length:
                raise ValueError()
        except (OSError,ValueError,http.client.HTTPException,UnicodeError):
            if not started:
                self.send_error(400,'Relay request rejected')
        finally:
            watch_stop.set()
            if watcher is not None:
                watcher.join(timeout=1)
            if upstream is not None:
                upstream.close()

    def _artifact(self,body):
        # Synthetic byte sink only. Production publication uses ArtifactStore
        # behind the authenticated gateway, never this optional fixture.
        with self.server.artifact_lock:
            root=self.server.artifact_root
            if len(list(root.iterdir()))>=8:
                raise ValueError()
            object_id=uuid.uuid4().hex
            fd=os.open(root/object_id,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as output:
                output.write(body)
            payload=json.dumps({'id':object_id,'size':len(body),'sha256':hashlib.sha256(body).hexdigest()}).encode()
        self.send_response(201)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(payload)))
        self.send_header('Connection','close')
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket',required=True)
    parser.add_argument('--upstream-port',required=True,type=int)
    parser.add_argument('--synthetic-artifacts',type=Path)
    args=parser.parse_args()
    if os.geteuid()!=0 or socket.gethostname()!='synthetic-execution-worker':
        parser.error('Run only as root inside the clean synthetic worker')
    if not re.fullmatch('/srv/jailer/firecracker/[a-f0-9]{32}/root/gateway.vsock_800[01]',args.socket) or not 1024<=args.upstream_port<=65535:
        parser.error('Invalid fixed synthetic socket or loopback port')
    if args.synthetic_artifacts:
        args.synthetic_artifacts.mkdir(mode=0o700,parents=True,exist_ok=True)
        info=args.synthetic_artifacts.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_mode&0o777!=0o700:
            parser.error('Artifact directory must be root-owned and private')
    with Relay(args.socket,args.upstream_port,artifact_root=args.synthetic_artifacts) as relay:
        os.chown(args.socket,65534,65534)
        relay.serve_forever(poll_interval=0.1)


if __name__=='__main__':
    main()
