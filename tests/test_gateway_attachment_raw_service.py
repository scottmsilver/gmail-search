import asyncio
from dataclasses import replace

import pytest

from gmail_search.gateway.attachment_raw_service import RunRawAttachmentService
from gmail_search.gateway.data_admission import DataAdmission
from gmail_search.gateway.registry import AccessDenied
from test_gateway_attachment_raw_http import setup as setup


@pytest.mark.parametrize('options',[dict(timeout_seconds=31),dict(timeout_seconds=True),dict(timeout_seconds=float('nan')),
    dict(watch_interval=.06),dict(watch_interval=0),dict(admission=DataAdmission(global_concurrency=3,owner_concurrency=1)),
    dict(admission=DataAdmission(global_concurrency=2,owner_concurrency=2))])
def test_configuration_caps_are_trusted_and_bounded(setup,options):
    values=dict(admission=setup[4]); values.update(options)
    with pytest.raises(ValueError): RunRawAttachmentService(setup[1],setup[3],**values)


@pytest.mark.asyncio
@pytest.mark.parametrize('aid',[True,0,-1,2**63,'1'])
async def test_bad_ids_never_load_or_publish(setup,aid):
    async def publish(source,check): pytest.fail('must not publish')
    with pytest.raises(ValueError):
        await setup[-1].deliver(setup[2],aid,deadline=asyncio.get_running_loop().time()+1,publish=publish)
    assert not setup[3].calls and setup[4].active=={}


@pytest.mark.asyncio
async def test_changed_immutable_run_binding_before_load_is_denied(setup,monkeypatch):
    service=setup[-1]; original=service.authorize; count=0
    async def authorize(token):
        nonlocal count
        count+=1
        lease=await original(token)
        return replace(lease,conversation_id='other') if count>1 else lease
    monkeypatch.setattr(service,'authorize',authorize)
    async def publish(source,check): pytest.fail('must not publish')
    with pytest.raises(AccessDenied):
        await service.deliver(setup[2],1,deadline=asyncio.get_running_loop().time()+1,publish=publish)
    assert not setup[3].calls and setup[4].active=={}


@pytest.mark.asyncio
async def test_lease_renewal_does_not_change_binding(setup):
    service=setup[-1]
    async def publish(source,check):
        lease=await service.authorize(setup[2])
        setup[0].heartbeat(lease.run_id,ttl=120)
        assert await check() is True
    await service.deliver(setup[2],1,deadline=asyncio.get_running_loop().time()+1,publish=publish)
    assert setup[4].active=={}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['invalid','publisher'])
async def test_failed_payload_frames_do_not_outlive_capacity(setup,failure):
    import gc
    import weakref
    from gmail_search.gateway.attachment_source import RawAttachmentInput
    observed=[]
    async def load(owner,aid):
        source=RawAttachmentInput(owner,aid,'bad' if failure=='invalid' else 'text/plain',b'x'*1_000_000)
        observed.append(weakref.ref(source))
        return source
    setup[3].load_raw=load
    async def publish(source,check):
        raise OSError('publisher error')
    try:
        await setup[-1].deliver(setup[2],1,deadline=asyncio.get_running_loop().time()+1,publish=publish)
    except RuntimeError:
        gc.collect()
        assert setup[4].active=={} and observed[0]() is None
    else:
        pytest.fail('Expected refusal')
