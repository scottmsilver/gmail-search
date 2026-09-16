"""Byte uploads never let guest paths choose a host file."""
import asyncio

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import Registry, AccessDenied
from gmail_search.gateway.artifacts import ArtifactStore


@pytest.fixture
def storage(tmp_path):
    active = {'alice', 'bob'}
    registry = Registry(tmp_path / 'runs.sqlite', is_active=lambda owner: owner in active)
    caps = Capabilities(registry)
    objects = tmp_path / 'objects'
    objects.mkdir(mode=0o700)
    return ArtifactStore(objects, caps, max_object_bytes=8, max_owner_bytes=12), registry, caps, active


def token(registry, caps, owner='alice', key='first'):
    run = registry.start_run(owner, 'conversation', request_key=key)
    return caps.issue(run.run_id, audience='artifact', operations={'artifact.commit'}).secret, run


async def chunks(*values):
    for value in values:
        yield value


def test_bytes_publish_survive_restart_and_remain_owner_scoped(storage):
    store, registry, caps, _ = storage
    secret, run = token(registry, caps)
    item = asyncio.run(store.publish(secret, chunks(b'hello'), filename='report.html'))
    assert item.filename == 'report.html'
    assert item.size == 5
    registry.finish(run.run_id, status='completed')
    restarted = ArtifactStore(store.root, caps, max_object_bytes=8, max_owner_bytes=12)
    assert restarted.read('alice', 'conversation', item.id) == b'hello'
    for owner, conversation in [('bob', 'conversation'), ('alice', 'other')]:
        with pytest.raises(AccessDenied):
            restarted.read(owner, conversation, item.id)
    assert item.download_headers['Content-Type'] == 'application/octet-stream'
    assert item.download_headers['Content-Disposition'].startswith('attachment;')
    assert item.download_headers['X-Content-Type-Options'] == 'nosniff'


@pytest.mark.parametrize('filename', ['../secret', '/etc/passwd', r'a\b', '.', '..', 'bad\r\nheader', '', 'a' * 201, '\ud800'])
def test_guest_paths_rejected_before_upload(storage, filename):
    store, registry, caps, _ = storage
    secret, _ = token(registry, caps)
    with pytest.raises(AccessDenied):
        asyncio.run(store.publish(secret, chunks(b'hi'), filename=filename))
    assert not list(store.root.iterdir())


def test_stream_quotas_and_owner_quota_leave_no_partial_files(storage):
    store, registry, caps, _ = storage
    secret, _ = token(registry, caps)
    with pytest.raises(AccessDenied):
        asyncio.run(store.publish(secret, chunks(b'12345678', b'9'), filename='x'))
    assert not list(store.root.iterdir())
    first = asyncio.run(store.publish(secret, chunks(b'12345678'), filename='one'))
    with pytest.raises(AccessDenied):
        asyncio.run(store.publish(secret, chunks(b'12345'), filename='two'))
    assert store.read('alice', 'conversation', first.id) == b'12345678'
    assert len(list(store.root.iterdir())) == 1


@pytest.mark.parametrize('change', ['cancel', 'revoke', 'fence'])
def test_midstream_revocation_and_stale_fence_abort_publication(storage, change):
    store, registry, caps, active = storage
    secret, run = token(registry, caps)
    async def changing():
        yield b'one'
        if change == 'revoke':
            active.remove('alice')
        elif change == 'cancel':
            registry.cancel(run.run_id)
        else:
            registry.finish(run.run_id, status='failed')
            registry.start_run('alice', 'conversation', request_key='new')
        yield b'two'
    with pytest.raises(AccessDenied):
        asyncio.run(store.publish(secret, changing(), filename='x'))
    assert not list(store.root.iterdir())


def test_symlink_object_never_reads_target(storage, tmp_path):
    store, registry, caps, _ = storage
    secret, _ = token(registry, caps)
    item = asyncio.run(store.publish(secret, chunks(b'ok'), filename='x'))
    target = tmp_path / 'secret'
    target.write_bytes(b'secret')
    obj = next(store.root.iterdir())
    obj.unlink()
    obj.symlink_to(target)
    with pytest.raises(AccessDenied):
        store.read('alice', 'conversation', item.id)


def test_stalled_stream_checks_revocation_without_waiting_for_body(storage):
    store, registry, caps, active = storage
    secret, _ = token(registry, caps)
    async def run():
        waiting = asyncio.Event()
        async def stalled():
            yield b'a'
            waiting.set()
            await asyncio.sleep(10)
        task = asyncio.create_task(store.publish(secret, stalled(), filename='x'))
        await waiting.wait()
        active.remove('alice')
        with pytest.raises(AccessDenied):
            await asyncio.wait_for(task, 1)
    asyncio.run(run())
    assert not list(store.root.iterdir())


def test_pending_reservations_bound_concurrent_uploads(storage):
    store, registry, caps, _ = storage
    secret, _ = token(registry, caps)
    async def run():
        ready, release = asyncio.Event(), asyncio.Event()
        async def held():
            yield b'12345678'
            ready.set()
            await release.wait()
        first = asyncio.create_task(store.publish(secret, held(), filename='one'))
        await ready.wait()
        with pytest.raises(AccessDenied):
            await store.publish(secret, chunks(b'12345'), filename='two')
        release.set()
        return await first
    item = asyncio.run(run())
    assert store.read('alice', 'conversation', item.id) == b'12345678'
    assert len(list(store.root.iterdir())) == 1


def test_interrupted_upload_is_reaped_after_restart(storage):
    store, registry, caps, _ = storage
    secret, run = token(registry, caps)
    item = asyncio.run(store.publish(secret, chunks(b'good'), filename='keep'))
    # Recreate the durable state of a process killed between file write and commit.
    pending_id = 'a' * 32
    (store.root / pending_id).write_bytes(b'partial')
    with registry._transaction() as db:
        db.execute('INSERT INTO artifacts(id,owner_id,conversation_id,run_id,filename,size,expires) VALUES(?,?,?,?,?,?,?)',
                   (pending_id, 'alice', 'conversation', run.run_id, 'pending', 8, 0))
    restarted = ArtifactStore(store.root, caps, max_object_bytes=8, max_owner_bytes=12)
    assert restarted.reap_pending() == 1
    assert not (store.root / pending_id).exists()
    assert restarted.read('alice', 'conversation', item.id) == b'good'
    assert restarted.reap_pending() == 0
