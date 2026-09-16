"""The real gateway factory must keep raw payload ownership outside buffering."""
import asyncio

import pytest

from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.service import RunQueryService
from test_gateway_attachment_raw_http import setup as raw_setup_fixture, Connection, packet


raw_setup = raw_setup_fixture


@pytest.mark.asyncio
async def test_raw_factory_route_is_opt_in(raw_setup):
    _, caps, token, source, admission, raw = raw_setup
    sql = RunQueryService(caps, None)
    conn = Connection(token)
    app = create_gateway_app(sql)
    await app(conn.scope, conn.receive, conn.send)
    assert conn.sent[0]['status'] == 404 and not source.calls
    app = create_gateway_app(sql, raw_attachments=raw)
    assert not source.calls and not admission.active
    conn = Connection(token)
    await app(conn.scope, conn.receive, conn.send)
    _, header, data = packet(conn)
    assert header['attachment_id'] == 1 and data == b'synthetic'
    assert not admission.active


@pytest.mark.asyncio
async def test_factory_keeps_raw_capacity_through_outer_send_cleanup(raw_setup):
    _, caps, token, source, admission, raw = raw_setup
    app = create_gateway_app(RunQueryService(caps, None), raw_attachments=raw)
    conn = Connection(token)
    conn.block_type = 'http.response.body'
    conn.send_gate = asyncio.Event()
    task = asyncio.create_task(app(conn.scope, conn.receive, conn.send))
    try:
        await asyncio.wait_for(conn.sending.wait(), 2)
        assert admission.active == {'alice': 1}
        task.cancel()
        await asyncio.wait_for(conn.send_closing.wait(), 2)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done() and admission.active == {'alice': 1}
        conn.send_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert conn.send_closed and not admission.active
    finally:
        conn.send_gate.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
