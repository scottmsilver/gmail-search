"""Synthetic worker relay reconstructs framing and never follows guest routing."""
import importlib.util
import http.server
from pathlib import Path
import socket
import threading

import pytest

spec=importlib.util.spec_from_file_location('vsock_relay',Path(__file__).parents[1]/'deploy/public/worker/vsock_http_relay.py')
relay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


@pytest.fixture
def server(tmp_path):
    received=[]
    raw_options={}
    class Upstream(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path,dict(self.headers),self.rfile.read(int(self.headers['Content-Length']))))
            if self.path == '/v1/attachment/raw':
                payload = raw_options.get('payload', b'framed-synthetic-bytes')
                self.send_response(raw_options.get('status', 200))
                headers = raw_options.get('headers', [('Content-Type','application/octet-stream'), ('Content-Length',str(len(payload)))])
                for name, value in headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.startswith('/v1/artifacts?'):
                payload = b'{"id":"artifact-id","filename":"result.txt","size":2}'
                self.send_response(201)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(payload)))
                self.send_header('Location','/v1/artifacts/artifact-id')
                self.send_header('Set-Cookie','upstream-session=secret')
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length','2')
            self.end_headers()
            self.wfile.write(b'{}')
        def log_message(self,*args): pass
    upstream=http.server.ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    proxy=relay.Relay(tmp_path/'relay.sock',upstream.server_port)
    threads=[threading.Thread(target=s.serve_forever,kwargs={'poll_interval':.01},daemon=True) for s in (upstream,proxy)]
    for t in threads: t.start()
    try: yield tmp_path/'relay.sock',received,raw_options
    finally:
        for s in (proxy,upstream): s.shutdown(); s.server_close()
        for t in threads: t.join(2)


def request(server,path='/v1/messages?beta=true',extra=b'',body=b'{}',base=True):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(3)
        sock.connect(str(server[0]))
        headers=b'x-api-key: synthetic\r\nContent-Length: '+str(len(body)).encode()+b'\r\nContent-Type: application/json\r\n' if base else b''
        sock.sendall(b'POST '+path.encode()+b' HTTP/1.1\r\nHost: evil.example\r\n'+headers+extra+b'\r\n'+body)
        chunks=[]
        while chunk:=sock.recv(65536): chunks.append(chunk)
        return b''.join(chunks)


def test_fixed_target_and_only_normalized_auth_forwarded(server):
    result=request(server,extra=b'X-User-ID: bob\r\nProxy-Authorization: secret\r\n')
    assert result.startswith(b'HTTP/1.1 200') and result.endswith(b'{}')
    path,headers,body=server[1][0]
    assert path=='/v1/messages' and body==b'{}'
    assert headers['Authorization']=='Bearer synthetic'
    assert headers['Host'].startswith('127.0.0.1:')
    assert 'X-User-ID' not in headers and 'Proxy-Authorization' not in headers


def test_typed_thread_route_only_forwards_exact_path(server):
    assert request(server, '/v1/thread', body=b'{"thread_id":"shared"}').startswith(b'HTTP/1.1 200')
    assert server[1][0][0] == '/v1/thread'
    server[1].clear()
    for path in ('/v1/thread?owner=bob', '/v1/thread/other', '/v1/%74hread'):
        assert request(server, path).startswith(b'HTTP/1.1 400')
    assert not server[1]


def test_attachment_parser_route_has_no_path_or_owner_query(server):
    assert request(server, '/v1/attachment/parse', body=b'{"attachment_id":1}').startswith(b'HTTP/1.1 200')
    assert server[1][0][0] == '/v1/attachment/parse'
    server[1].clear()
    for path in ('/v1/attachment/parse?owner=bob', '/v1/attachment/parse?path=/secret', '/v1/attachment/raw?path=/secret'):
        assert request(server, path).startswith(b'HTTP/1.1 400')
    assert not server[1]


@pytest.mark.parametrize('path,extra',[
    ('http://evil.example/v1/messages',b''),('/admin',b''),('/v1/messages?owner=bob',b''),
    ('/v1/messages',b'Content-Length: 2\r\n'),('/v1/messages',b'Transfer-Encoding: chunked\r\n'),
    ('/v1/messages',b'Cookie: session=other\r\n'),('/v1/messages',b'Authorization: Bearer duplicate\r\n')])
def test_rejects_smuggling_and_routing_overrides(server,path,extra):
    assert request(server,path,extra).startswith(b'HTTP/1.1 400')
    assert not server[1]


def test_artifact_upload_relays_gateway_201_without_upstream_response_headers(server):
    artifact_headers=b'x-api-key: synthetic\r\nContent-Length: 2\r\nContent-Type: application/octet-stream\r\n'
    result=request(server,'/v1/artifacts?filename=result.txt',artifact_headers,b'42',False)
    assert result.startswith(b'HTTP/1.1 201')
    assert result.endswith(b'{"id":"artifact-id","filename":"result.txt","size":2}')
    assert b'Location:' not in result and b'Set-Cookie:' not in result
    path, upstream_headers, body = server[1][0]
    assert path == '/v1/artifacts?filename=result.txt'
    assert upstream_headers['Authorization'] == 'Bearer synthetic' and body == b'42'
    assert request(server,'/v1/artifacts?filename=../secret',artifact_headers,b'42',False).startswith(b'HTTP/1.1 400')


def test_body_limit_rejects_before_waiting_for_upload(server):
    headers=b'x-api-key: synthetic\r\nContent-Length: 99999999\r\nContent-Type: application/json\r\n'
    assert request(server,extra=headers,body=b'',base=False).startswith(b'HTTP/1.1 400')
    assert not server[1]


def test_gemini_fixed_route_normalizes_google_capability_header(server):
    path = '/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse'
    headers = b'x-goog-api-key: synthetic\r\nContent-Length: 2\r\nContent-Type: application/json\r\n'
    assert request(server, path, headers, base=False).startswith(b'HTTP/1.1 200')
    forwarded, fields, body = server[1][0]
    assert forwarded == path and body == b'{}'
    assert fields['Authorization'] == 'Bearer synthetic'
    assert 'x-goog-api-key' not in fields
    server[1].clear()
    for invalid in (path.replace('3.8-flash', 'other'), path.replace('?alt=sse', ''),
                    path + '&alt=sse', path + '&key=secret', path.replace('alt=sse', 'alt=json')):
        assert request(server, invalid, headers, base=False).startswith(b'HTTP/1.1 400')
    assert not server[1]


@pytest.mark.parametrize('extra', [b'x-goog-api-key: second\r\n',
                                  b'Authorization: Bearer second\r\nx-goog-api-key: third\r\n'])
def test_google_alias_does_not_allow_multiple_credentials(server, extra):
    assert request(server, extra=extra).startswith(b'HTTP/1.1 400')
    assert not server[1]


def test_event_submission_route_forwards_no_owner_or_replay_selector(server):
    assert request(server, '/v1/events', body=b'{"type":"text","text":"synthetic"}').startswith(b'HTTP/1.1 200')
    assert server[1][0][0] == '/v1/events'
    server[1].clear()
    for path in ('/v1/events?owner=bob', '/v1/events?after=1', '/v1/events/other'):
        assert request(server, path).startswith(b'HTTP/1.1 400')
    assert not server[1]


@pytest.mark.parametrize('send_headers', [False, True], ids=['before-headers', 'during-body'])
def test_guest_disconnect_closes_stalled_upstream(tmp_path, send_headers):
    started, disconnected, release = threading.Event(), threading.Event(), threading.Event()
    class WaitingUpstream(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.connection.settimeout(.05)
            if send_headers:
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                self.wfile.write(b'{')
                self.wfile.flush()
            started.set()
            while not release.is_set():
                try:
                    if not self.connection.recv(1):
                        disconnected.set()
                        return
                except socket.timeout:
                    continue
        def log_message(self, *args):
            pass
    upstream = http.server.ThreadingHTTPServer(('127.0.0.1',0), WaitingUpstream)
    proxy = relay.Relay(tmp_path/'cancel.sock', upstream.server_port)
    threads = [threading.Thread(target=service.serve_forever, kwargs={'poll_interval':.01}, daemon=True)
               for service in (upstream,proxy)]
    for thread in threads:
        thread.start()
    guest = socket.socket(socket.AF_UNIX)
    try:
        guest.connect(str(tmp_path/'cancel.sock'))
        guest.sendall(b'POST /v1/thread HTTP/1.1\r\nHost: guest\r\nx-api-key: synthetic\r\nContent-Length: 2\r\n\r\n{}')
        assert started.wait(2)
        guest.close()
        assert disconnected.wait(1), 'relay left upstream running after guest disconnected'
    finally:
        guest.close()
        release.set()
        for service in (proxy,upstream):
            service.shutdown()
            service.server_close()
        for thread in threads:
            thread.join(2)


def test_search_route_accepts_only_exact_private_worker_path(server):
    assert request(server,'/v1/search',body=b'{"query":"needle"}').startswith(b'HTTP/1.1 200')
    assert server[1][0][0]=='/v1/search'
    server[1].clear()
    for path in ('/v1/search?owner=bob','/v1/search/other','/v1/%73earch'):
        assert request(server,path).startswith(b'HTTP/1.1 400')
    assert not server[1]


def test_facts_route_accepts_only_exact_private_worker_path(server):
    assert request(server, '/v1/find-facts', body=b'{"query":"cars"}').startswith(b'HTTP/1.1 200')
    assert server[1][0][0] == '/v1/find-facts'
    for path in ('/v1/find-facts?owner=bob', '/v1/find-facts/other', '/v1/find%2dfacts'):
        assert not request(server, path, body=b'{}').startswith(b'HTTP/1.1 200')
    assert len(server[1]) == 1


def test_metadata_route_accepts_only_exact_private_worker_path(server):
    assert request(server, '/v1/query-emails', body=b'{"sender":"example"}').startswith(b'HTTP/1.1 200')
    assert server[1][0][0] == '/v1/query-emails'
    for path in ('/v1/query-emails?owner=bob', '/v1/query-emails/other', '/v1/query%2demails'):
        assert not request(server, path, body=b'{}').startswith(b'HTTP/1.1 200')
    assert len(server[1]) == 1


@pytest.mark.parametrize('path', ['/v1/attachment/meta', '/v1/attachment/text', '/v1/attachment/list'])
def test_attachment_json_routes_are_exact(server, path):
    assert request(server, path).startswith(b'HTTP/1.1 200')
    assert server[1][0][0] == path
    for suffix in ('?owner=bob', '/other'):
        assert request(server, path+suffix).startswith(b'HTTP/1.1 400')
    assert len(server[1]) == 1


def test_raw_route_preserves_validated_binary_length(server):
    result = request(server, '/v1/attachment/raw', body=b'{"attachment_id":1}')
    assert result.startswith(b'HTTP/1.1 200')
    headers, payload = result.split(b'\r\n\r\n', 1)
    assert b'Content-Type: application/octet-stream' in headers
    assert b'Content-Length: '+str(len(payload)).encode() in headers
    assert payload == b'framed-synthetic-bytes'
    for path in ('/v1/attachment/raw?owner=bob', '/v1/attachment/%72aw', '/v1/attachment/raw/other'):
        assert request(server, path).startswith(b'HTTP/1.1 400')


@pytest.mark.parametrize('headers', [
    [('Content-Type','application/octet-stream')],
    [('Content-Type','application/json'),('Content-Length','2')],
    [('Content-Type','application/octet-stream'),('Content-Length','2'),('Content-Length','2')],
    [('Content-Type','application/octet-stream'),('Content-Length','10489861')],
    [('Content-Type','application/octet-stream'),('Content-Length','20'),('Content-Encoding','gzip')],
    [('Content-Type','application/octet-stream'),('Content-Length','20'),('Transfer-Encoding','chunked')],
])
def test_raw_relay_refuses_invalid_upstream_framing_before_success(server, headers):
    server[2]['headers'] = headers
    assert request(server, '/v1/attachment/raw').startswith(b'HTTP/1.1 400')


def test_raw_request_limit_refuses_before_upload(server):
    headers=b'x-api-key: synthetic\r\nContent-Length: 4097\r\nContent-Type: application/json\r\n'
    assert request(server, '/v1/attachment/raw', extra=headers, body=b'', base=False).startswith(b'HTTP/1.1 400')
    assert not server[1]


@pytest.mark.parametrize('value',['020','+20','20 ','-20','0','4','20,20','20\t'])
def test_additional_noncanonical_or_short_lengths(server,value):
    server[2]['headers']=[('Content-Type','application/octet-stream'),('Content-Length',value)]
    result=request(server,'/v1/attachment/raw')
    assert result.startswith(b'HTTP/1.1 400')
    assert b'framed-synthetic-bytes' not in result

@pytest.mark.parametrize('status',[201,204,206,301,403,500])
def test_only_200_accepted(server,status):
    server[2]['status']=status
    assert request(server,'/v1/attachment/raw').startswith(b'HTTP/1.1 400')

@pytest.mark.parametrize('headers',[
 [('Content-Type','application/octet-stream'),('Content-Type','application/octet-stream'),('Content-Length','20')],
 [('Content-Type','application/octet-stream; charset=binary'),('Content-Length','20')],
 [('Content-Type','application/octet-stream'),('Content-Length','20'),('Content-Encoding','identity')],
 [('Content-Type','application/octet-stream'),('Content-Length','20'),('Transfer-Encoding','identity')],
])
def test_raw_mime_and_encoding_closed(server,headers):
    server[2]['headers']=headers
    assert request(server,'/v1/attachment/raw').startswith(b'HTTP/1.1 400')

@pytest.mark.parametrize('payload',[b'',b'12345',b'\x00\xff\r\n\x00\x7f'])
def test_truncated_binary_response_closes_without_second_response(server,payload):
    server[2]['payload']=payload
    server[2]['headers']=[('Content-Type','application/octet-stream'),('Content-Length','20')]
    result=request(server,'/v1/attachment/raw')
    headers,body=result.split(b'\r\n\r\n',1)
    assert headers.startswith(b'HTTP/1.1 200')
    assert b'Content-Length: 20' in headers and body==payload
    assert result.count(b'HTTP/1.1')==1 and b'Relay request rejected' not in result
