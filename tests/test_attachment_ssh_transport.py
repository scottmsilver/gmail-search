"""Local subprocess tests only; no SSH server or private credentials involved."""
import sys
import threading
import time
import uuid

import pytest

from gmail_search.gateway.attachment_rpc import context_digest, response_header, encode_frame


def request():
    context = dict(owner_id='alice', run_id='run', conversation_id='conversation', fence=1,
                   attachment_id=1, deadline=time.time()+40)
    return dict(version=1, op='stop', request_id=str(uuid.uuid4()), job_id=str(uuid.uuid4()),
                context_sha256=context_digest(context), payload_size=0)


def test_ssh_command_is_fixed_and_has_no_inherited_routing(tmp_path):
    from gmail_search.gateway.attachment_remote import SSHTransport
    key, hosts = tmp_path / 'key', tmp_path / 'known_hosts'
    key.write_text('synthetic key, never used')
    key.chmod(0o600)
    hosts.write_text('synthetic host, never used')
    hosts.chmod(0o600)
    transport = SSHTransport(host='127.0.0.1', port=22092, private_key=key, known_hosts=hosts)
    command = transport.command()
    assert command[:3] == ['/usr/bin/ssh', '-F', '/dev/null']
    assert command[-2:] == ['gmail-attachment-rpc@127.0.0.1', 'attachment-rpc-v1']
    for option in ('IdentityAgent=none', 'ForwardAgent=no', 'ClearAllForwardings=yes',
                   'StrictHostKeyChecking=yes', 'ProxyCommand=none', 'ProxyJump=none'):
        assert option in command
    with pytest.raises(ValueError):
        SSHTransport(host='host; echo secret', port=22, private_key=key, known_hosts=hosts)


def test_process_exchange_is_bounded_and_drains_stderr():
    from gmail_search.gateway.attachment_remote import exchange_process
    packet = b'synthetic-input'
    result = exchange_process([sys.executable, '-c', 'import sys; b=sys.stdin.buffer.read(); sys.stderr.write("diagnostic"); sys.stdout.buffer.write(b[::-1])'], packet,
                              threading.Event(), timeout=2, max_output=100)
    assert result == packet[::-1]


@pytest.mark.parametrize('code', ['import sys; sys.stdout.buffer.write(b"x"*100000)',
                                 'import sys; sys.stderr.buffer.write(b"x"*100000)',
                                 'import time; time.sleep(30)'])
def test_process_limits_kill_and_reap(code):
    from gmail_search.gateway.attachment_remote import exchange_process, RemoteAttachmentUnavailable
    before = time.monotonic()
    with pytest.raises(RemoteAttachmentUnavailable):
        exchange_process([sys.executable, '-c', code], b'', threading.Event(), timeout=.2, max_output=100)
    assert time.monotonic()-before < 2


def test_response_binding_rejects_foreign_or_trailing_bytes():
    from gmail_search.gateway.attachment_remote import decode_response, RemoteAttachmentUnavailable
    header = request()
    response = response_header(header, 'stopped')
    assert decode_response(header, encode_frame(response, response=True))[0]['status'] == 'stopped'
    foreign = dict(response, job_id=str(uuid.uuid4()))
    for wire in (encode_frame(foreign, response=True), encode_frame(response, response=True)+b'junk'):
        with pytest.raises(RemoteAttachmentUnavailable):
            decode_response(header, wire)
