"""Closed, bounded attachment RPC framing (stdlib only; copied to worker).

Readers supply a blocking ``read(n)`` callback with an external absolute read
 deadline. No protocol field selects a path, executable or worker resource limit.
"""
import hashlib
import json
import math
import re
import struct
import uuid

VERSION = 1
MAX_HEADER = 4096
MAX_INPUT = 10 * 1024**2
MAX_OUTPUT = 8 * 1024**2
MIMES = frozenset({'application/pdf', 'application/zip', 'image/png', 'image/jpeg',
                   'image/gif', 'image/webp', 'image/tiff'})
COMMON = {'version', 'op', 'request_id', 'job_id', 'context_sha256', 'payload_size'}
CONTEXT = {'owner_id', 'run_id', 'conversation_id', 'fence', 'attachment_id', 'deadline'}


class ProtocolError(ValueError):
    pass


def _require(ok):
    if not ok:
        raise ProtocolError('Invalid attachment RPC')


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                          allow_nan=False).encode('ascii')
    except (ValueError, TypeError, RecursionError):
        raise ProtocolError('Invalid attachment RPC') from None


def _context(context):
    _require(type(context) is dict and set(context) == CONTEXT)
    for key in ('owner_id', 'run_id', 'conversation_id'):
        value = context[key]
        _require(type(value) is str and 0 < len(value) <= 200
                 and all(ord(c) >= 32 and ord(c) != 127 for c in value))
    _require(type(context['fence']) is int and 0 <= context['fence'] < 2**63)
    _require(type(context['attachment_id']) is int and 0 < context['attachment_id'] < 2**63)
    deadline = context['deadline']
    _require(type(deadline) in (int, float) and 0 < deadline < 2**53 and math.isfinite(deadline))


def context_digest(context):
    _context(context)
    return hashlib.sha256(_json(context)).hexdigest()


def _digest(value):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None)


def _header(header, response=False):
    _require(type(header) is dict)
    _require(type(header.get('version')) is int and header['version'] == VERSION)
    _require(type(header.get('op')) is str and header['op'] in ('start', 'poll', 'stop'))
    extra = ({'status', 'code', 'payload_sha256'} if response else
             {'context', 'mime_type', 'options', 'input_sha256'} if header['op'] == 'start' else
             {'renew_seq'} if header['op'] == 'poll' else set())
    _require(set(header) == COMMON | extra)
    for key in ('job_id', 'request_id'):
        value = header[key]
        _require(type(value) is str)
        try:
            _require(str(uuid.UUID(value)) == value)
        except (ValueError, AttributeError):
            raise ProtocolError('Invalid attachment RPC') from None
    _digest(header['context_sha256'])
    size = header['payload_size']
    _require(type(size) is int and 0 <= size <= (MAX_OUTPUT if response else MAX_INPUT))
    if response:
        _require(header['status'] in ('running', 'done', 'stopped', 'error'))
        _require(type(header['code']) is str and re.fullmatch('[a-z_]{1,40}', header['code']) is not None)
        _digest(header['payload_sha256'])
        _require(header['status'] == 'done' or size == 0)
    elif header['op'] == 'start':
        _require(size > 0)
        _require(context_digest(header['context']) == header['context_sha256'])
        _require(type(header['mime_type']) is str and header['mime_type'] in MIMES)
        options = header['options']
        _require(type(options) is dict and set(options) == {'dpi', 'pages'})
        _require(type(options['dpi']) is int and 72 <= options['dpi'] <= 200)
        pages = options['pages']
        _require(type(pages) is list and len(pages) <= 8)
        _require(all(type(p) is int and 1 <= p <= 100000 for p in pages))
        _require(len(set(pages)) == len(pages))
        _digest(header['input_sha256'])
    else:
        _require(size == 0)
        if header['op'] == 'poll':
            _require(type(header['renew_seq']) is int and 0 < header['renew_seq'] < 2**63)


def validate_request(header, payload):
    _validate(header, payload, False)


def _validate(header, payload, response):
    _header(header, response)
    _require(type(payload) is bytes and len(payload) == header['payload_size'])
    key = 'payload_sha256' if response else 'input_sha256'
    if response or header['op'] == 'start':
        _require(hashlib.sha256(payload).hexdigest() == header[key])
    raw = _json(header)
    _require(0 < len(raw) <= MAX_HEADER)
    if response:
        _require(4 + len(raw) + len(payload) <= MAX_OUTPUT)
    return raw


def encode_frame(header, payload=b'', *, response=False):
    raw = _validate(header, payload, response)
    return struct.pack('!I', len(raw)) + raw + payload


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result)
        result[key] = value
    return result


def read_frame(read, *, response=False):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            part = read(min(size - len(data), 65536))
            _require(type(part) is bytes and 0 < len(part) <= size - len(data))
            data.extend(part)
        return bytes(data)
    size = struct.unpack('!I', exact(4))[0]
    _require(0 < size <= MAX_HEADER)
    try:
        header = json.loads(exact(size).decode('utf-8'), object_pairs_hook=_pairs,
                            parse_constant=lambda _: _require(False))
    except (ValueError, UnicodeError, RecursionError):
        raise ProtocolError('Invalid attachment RPC') from None
    _header(header, response)
    if response:
        _require(4 + size + header['payload_size'] <= MAX_OUTPUT)
    payload = exact(header['payload_size'])
    _validate(header, payload, response)
    return header, payload


def response_header(request, status, *, code='ok', payload=b''):
    result = {key: request[key] for key in COMMON}
    result.update(status=status, code=code, payload_size=len(payload),
                  payload_sha256=hashlib.sha256(payload).hexdigest())
    return result
