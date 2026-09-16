"""Actual gateway packet encoder -> socket relay -> private guest file.

Synthetic bytes only. Authentication/database ownership have separate factory
integration tests; this checks the binary contract across the worker boundary.
"""
import asyncio
import hashlib
import os

import pytest

from gmail_search.gateway.attachment_raw_http import _publish
from gmail_search.gateway.attachment_source import RawAttachmentInput
from test_guest_attachment_download import modules
from test_worker_vsock_relay import server as _relay_server

relay_server = _relay_server


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', [None, 'truncated', 'corrupt', 'wrong_attachment'])
async def test_gateway_packet_through_relay_to_guest(relay_server, tmp_path, monkeypatch, fault):
    transport, download = modules()
    data = b'synthetic opaque attachment\x00\xff' * 3000
    events = []

    async def check():
        return True

    async def send(event):
        events.append(event)

    source = RawAttachmentInput('alice', 1, 'application/octet-stream', data)
    await _publish(source, check, send)
    packet = b''.join(event.get('body', b'') for event in events)
    headers = [(key.decode('ascii'), value.decode('ascii')) for key, value in events[0]['headers']]
    if fault == 'truncated':
        packet = packet[:-1]
    elif fault == 'corrupt':
        packet = packet[:-1] + bytes([packet[-1] ^ 1])
    relay_server[2].update(payload=packet, headers=headers)

    async def connect():
        return await asyncio.open_unix_connection(str(relay_server[0]), limit=16384)

    monkeypatch.setattr(transport, '_open', connect)
    work = tmp_path / 'work'
    work.mkdir(mode=0o700)
    fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
    slots = asyncio.Semaphore(2)
    client = download.GuestAttachmentDownloader(fd, 'a' * 64, slots=slots)
    try:
        result = await client.download_many(
            (2 if fault == 'wrong_attachment' else 1,),
            deadline=asyncio.get_running_loop().time() + 5,
        )
        files = [path for path in work.rglob('*') if path.is_file()]
        if fault:
            assert set(result[0]) == {'error'}
            assert files == []
            assert client.usage == {'files': 0, 'bytes': 0, 'quarantined': 0}
        else:
            assert set(result[0]) == {'relative_path', 'size_bytes', 'sha256'}
            path = work / result[0]['relative_path']
            assert files == [path]
            assert path.read_bytes() == data
            assert result[0]['size_bytes'] == len(data)
            assert result[0]['sha256'] == hashlib.sha256(data).hexdigest()
        assert slots._value == 2
        assert relay_server[1][0][0] == '/v1/attachment/raw'
    finally:
        await client.aclose()
        os.close(fd)
