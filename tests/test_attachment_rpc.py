"""Strict framing shared by controller, SSH frontend and privileged manager."""
import hashlib
import io
import json
import struct
import uuid

import pytest

from gmail_search.gateway import attachment_rpc as rpc


def request(op='start', *, job_id=None, context=None):
    context = context or dict(owner_id='alice', run_id='run', conversation_id='conversation',
                              fence=1, attachment_id=1, deadline=2000000000.0)
    h = dict(version=1, op=op, request_id=str(uuid.uuid4()), job_id=job_id or str(uuid.uuid4()),
             context_sha256=rpc.context_digest(context), payload_size=1 if op == 'start' else 0)
    if op == 'start':
        h.update(context=context, mime_type='application/pdf', options=dict(dpi=100, pages=[]),
                 input_sha256=hashlib.sha256(b'x').hexdigest())
    if op == 'poll':
        h['renew_seq'] = 1
    return h


def test_roundtrip():
    h = request()
    assert rpc.read_frame(io.BytesIO(rpc.encode_frame(h, b'x')).read) == (h, b'x')


@pytest.mark.parametrize('raw', [b'{"version":1,"version":1}', b'{"x":NaN}', b'[]'])
def test_strict_json(raw):
    with pytest.raises(rpc.ProtocolError):
        rpc.read_frame(io.BytesIO(struct.pack('!I', len(raw)) + raw).read)


def test_header_limit_rejected_before_reading_header():
    stream = io.BytesIO(struct.pack('!I', 4097) + b'never-read')
    with pytest.raises(rpc.ProtocolError):
        rpc.read_frame(stream.read)
    assert stream.tell() == 4


@pytest.mark.parametrize('field,value', [('payload_size', 10*1024**2+1), ('command', 'sh'),
                                       ('version', True), ('request_id', 'not-uuid')])
def test_bad_header_before_payload_read(field, value):
    h = request()
    h[field] = value
    raw = json.dumps(h).encode()
    stream = io.BytesIO(struct.pack('!I', len(raw)) + raw + b'x')
    with pytest.raises(rpc.ProtocolError):
        rpc.read_frame(stream.read)
    assert stream.tell() == 4 + len(raw)


def test_payload_digest_and_context_binding():
    with pytest.raises(rpc.ProtocolError):
        rpc.encode_frame(request(), b'y')
    h = request()
    h['context']['owner_id'] = 'bob'
    with pytest.raises(rpc.ProtocolError):
        rpc.encode_frame(h, b'x')


@pytest.mark.parametrize('change', [dict(upstream='https://x'), dict(dpi=True), dict(pages=[1, 1])])
def test_options_closed_schema(change):
    h = request()
    h['options'].update(change)
    with pytest.raises(rpc.ProtocolError):
        rpc.encode_frame(h, b'x')


def test_response_total_size_limit():
    h = request('poll')
    h.pop('renew_seq')
    payload = b'x' * rpc.MAX_OUTPUT
    h.update(status='done', code='ok', payload_size=len(payload),
             payload_sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(rpc.ProtocolError):
        rpc.encode_frame(h, payload, response=True)


@pytest.mark.parametrize('mutation', [
    lambda h: h.update(context=None),
    lambda h: h.update(context={'owner_id': 'bob'}),
    lambda h: h['context'].update(deadline=float('inf')),
    lambda h: h['context'].update(deadline=10**400),
    lambda h: h.update(options={'dpi': 100, 'pages': [{}]}),
    lambda h: h.update(op=[]),
])
def test_malformed_header_is_protocol_error(mutation):
    h = request()
    mutation(h)
    with pytest.raises(rpc.ProtocolError):
        rpc.encode_frame(h, b'x')
