"""Capability-bound JSON attachment reads; no filesystem or parser calls."""

import asyncio
from dataclasses import replace
import importlib

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.gateway.attachment_reader import OwnerAttachmentReader
from test_gateway_attachment_reader import Gateway, row


def compose(tmp_path, api=None, owner="alice", operations=frozenset({"meta", "text"})):
    module = importlib.import_module("gmail_search.gateway.attachment_read_service")
    registry = Registry(tmp_path / "attachment-read.sqlite", is_active=lambda _: True)
    caps = Capabilities(registry)
    run = registry.start_run(
        owner, "conversation", request_key="attachment-read", writer=False
    )
    token = caps.issue(run.run_id, audience="attachment", operations=operations)
    api = api or Gateway()
    return (
        module.RunAttachmentReadService(caps, OwnerAttachmentReader(api)),
        caps,
        token,
        api,
    )


@pytest.mark.asyncio
async def test_metadata_projection_preserves_fields_without_private_owner(tmp_path):
    service, _, token, api = compose(tmp_path)
    result = await service.describe(token.secret, attachment_id=1)
    assert result == dict(
        attachment_id=1,
        message_id="message",
        thread_id="thread",
        filename="file.pdf",
        mime_type="application/pdf",
        size_bytes=123,
        fetch_status="ok",
        text_chars=4,
        stored_text_state="present",
        extraction_complete=None,
        cite_ref="thread",
    )
    assert api.calls[0][0] == "alice"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chars,text,state,complete",
    [
        (None, None, "missing", False),
        (0, "", "empty", True),
        (4, "body", "present", True),
    ],
)
async def test_text_projection_preserves_missing_empty_and_present(
    tmp_path, chars, text, state, complete
):
    service, _, token, _ = compose(
        tmp_path, Gateway([row(text_chars=chars) + (text,)], text=True)
    )
    result = await service.text_page(token.secret, attachment_id=1)
    assert result["extracted_text"] == text and result["stored_text_state"] == state
    assert (
        result["stored_text_complete"] is complete
        and result["page_complete"] is complete
    )
    assert result["extraction_complete"] is None and result["next_offset"] is None
    assert "owner_id" not in result


@pytest.mark.asyncio
async def test_manifest_projection_preserves_cursor_and_legacy_id(tmp_path):
    service, _, token, _ = compose(tmp_path, Gateway([row(aid=1), row(aid=2)]))
    result = await service.list_for_thread(token.secret, thread_id="thread", limit=1)
    assert (
        result["attachments"][0]["id"] == 1
        and result["attachments"][0]["attachment_id"] == 1
    )
    assert (
        result["next_attachment_id"] == 1
        and not result["complete"]
        and not result["source_complete"]
    )
    assert "owner_id" not in result and all(
        "owner_id" not in item for item in result["attachments"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["meta", "text"])
async def test_operations_are_separate_and_do_not_query_when_denied(
    tmp_path, operation
):
    service, _, token, api = compose(tmp_path, operations={operation})
    with pytest.raises(AccessDenied):
        await (service.text_page if operation == "meta" else service.describe)(
            token.secret, attachment_id=1
        )
    assert not api.calls


@pytest.mark.asyncio
async def test_tampered_reader_owner_is_rejected_before_projection(tmp_path):
    service, _, token, _ = compose(tmp_path)
    original = service.reader.describe

    async def forged(*args, **kwargs):
        return replace(await original(*args, **kwargs), owner_id="bob")

    service.reader.describe = forged
    with pytest.raises(RuntimeError, match="^Attachment data is unavailable.$"):
        await service.describe(token.secret, attachment_id=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"attachment_id": True},
        {"attachment_id": 0},
        {"offset": -1},
        {"offset": 2147483647},
        {"limit": True},
        {"limit": 100001},
    ],
)
async def test_invalid_text_options_never_read(tmp_path, change):
    service, _, token, api = compose(tmp_path)
    with pytest.raises(ValueError):
        await service.text_page(token.secret, **{"attachment_id": 1, **change})
    assert not api.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["revoked", "audience"])
async def test_bad_capability_never_reads(tmp_path, kind):
    service, caps, token, api = compose(tmp_path)
    if kind == "revoked":
        caps.revoke(token.secret)
    else:
        lease = caps.authorize(token.secret, audience="attachment", operation="meta")
        token = caps.issue(lease.run_id, audience="retrieval", operations={"meta"})
    with pytest.raises(AccessDenied):
        await service.describe(token.secret, attachment_id=1)
    assert not api.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"stored_text_state": "complete"},
        {"extraction_complete": True},
        {"attachment_id": 2},
    ],
)
async def test_forged_metadata_state_or_identity_is_rejected(tmp_path, changes):
    service, _, token, _ = compose(tmp_path)
    original = service.reader.describe

    async def forged(*args, **kwargs):
        return replace(await original(*args, **kwargs), **changes)

    service.reader.describe = forged
    with pytest.raises(RuntimeError, match="^Attachment data is unavailable.$"):
        await service.describe(token.secret, attachment_id=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "manifest"])
async def test_forged_paging_flags_are_refused(tmp_path, kind):
    service, _, token, _ = compose(
        tmp_path,
        Gateway([row(text_chars=4) + ("body",)], text=True)
        if kind == "text"
        else Gateway([row(aid=1), row(aid=2)]),
    )
    name = "text_page" if kind == "text" else "list_for_thread"
    original = getattr(service.reader, name)

    async def forged(*args, **kwargs):
        page = await original(*args, **kwargs)
        return replace(
            page, **({"next_offset": 3} if kind == "text" else {"complete": True})
        )

    setattr(service.reader, name, forged)
    with pytest.raises(RuntimeError, match="^Attachment data is unavailable.$"):
        if kind == "text":
            await service.text_page(token.secret, attachment_id=1)
        else:
            await service.list_for_thread(token.secret, thread_id="thread", limit=1)


@pytest.mark.asyncio
async def test_incomplete_empty_manifest_preserves_unknown_inventory(tmp_path):
    service, _, token, _ = compose(tmp_path, Gateway([], complete=False))
    result = await service.list_for_thread(token.secret, thread_id="thread")
    assert (
        result["attachments"] == []
        and not result["complete"]
        and not result["source_complete"]
    )
    assert result["pagination_limited"] and result["next_attachment_id"] is None


@pytest.mark.asyncio
async def test_partial_text_is_stored_slice_and_not_extraction_completeness(tmp_path):
    service, _, token, _ = compose(
        tmp_path, Gateway([row(text_chars=100) + ("0123456789",)], text=True)
    )
    result = await service.text_page(token.secret, attachment_id=1, offset=20, limit=10)
    assert (
        result["offset"] == 20
        and result["next_offset"] == 30
        and result["total_chars"] == 100
    )
    assert result["page_complete"] and not result["stored_text_complete"]
    assert result["extraction_complete"] is None


@pytest.mark.asyncio
async def test_output_budget_fails_without_partial_projection(tmp_path, monkeypatch):
    module = importlib.import_module("gmail_search.gateway.attachment_read_service")
    service, _, token, _ = compose(tmp_path)
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 100)
    with pytest.raises(RuntimeError, match="^Attachment data is unavailable.$"):
        await service.describe(token.secret, attachment_id=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["revoke", "deadline"])
async def test_final_authorization_follows_public_projection(
    tmp_path, monkeypatch, failure
):
    module = importlib.import_module("gmail_search.gateway.attachment_read_service")
    service, caps, token, _ = compose(tmp_path)
    original = module._bounded
    loop = asyncio.get_running_loop()
    clock = loop.time

    def project(value):
        result = original(value)
        if failure == "revoke":
            caps.revoke(token.secret)
        else:
            monkeypatch.setattr(loop, "time", lambda: clock() + 31)
        return result

    monkeypatch.setattr(module, "_bounded", project)
    try:
        with pytest.raises(AccessDenied if failure == "revoke" else TimeoutError):
            await service.describe(token.secret, attachment_id=1)
    finally:
        monkeypatch.setattr(loop, "time", clock)


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_reader_cleanup(tmp_path):
    service, _, token, api = compose(tmp_path)
    started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def query(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()

    api.query = query
    task = asyncio.create_task(service.describe(token.secret, attachment_id=1))
    try:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        await asyncio.wait_for(closing.wait(), 1)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_revocation_during_reader_cleanup_prevents_response(tmp_path):
    service, caps, token, _ = compose(tmp_path)
    closing, release = asyncio.Event(), asyncio.Event()
    original = service.reader.describe

    async def slow(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        finally:
            closing.set()
            await release.wait()

    service.reader.describe = slow
    task = asyncio.create_task(service.describe(token.secret, attachment_id=1))
    try:
        await asyncio.wait_for(closing.wait(), 1)
        caps.revoke(token.secret)
        release.set()
        with pytest.raises(AccessDenied):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


from test_gateway_database_integration import database as database, reader_dsn


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_index", [0, 1])
async def test_real_owner_bound_metadata_text_and_manifest(
    database, tmp_path, owner_index
):
    import json
    import psycopg
    from gmail_search.gateway.database import (
        QueryGateway,
        ReaderCredential,
        ReaderRegistry,
    )
    from gmail_search.gateway.data_admission import DataAdmission

    dsn, owners = database
    with psycopg.connect(dsn, autocommit=True) as conn:
        for index, owner in enumerate(owners):
            conn.execute(
                "INSERT INTO messages(id,thread_id,user_id) VALUES(%s,%s,%s)",
                ("message", "thread", owner),
            )
            for aid, text in ((1, None if index == 0 else "private bob text"), (2, "")):
                conn.execute(
                    "INSERT INTO attachments(id,message_id,filename,mime_type,size_bytes,fetch_status,extracted_text,user_id) VALUES(%s,%s,%s,%s,0,%s,%s,%s)",
                    (
                        aid,
                        "message",
                        owner + ".txt",
                        "text/plain",
                        "fetch_failed",
                        text,
                        owner,
                    ),
                )
    owner = owners[owner_index]
    capacity = DataAdmission(global_concurrency=1, owner_concurrency=1)
    api = QueryGateway(
        ReaderRegistry(
            {owner: ReaderCredential(owner, reader_dsn(dsn, owner))},
            is_active=lambda _: True,
        ),
        admission=capacity,
    )
    service, _, token, _ = compose(tmp_path, api, owner)
    meta = await service.describe(token.secret, attachment_id=1)
    assert (
        meta["filename"] == owner + ".txt"
        and meta["size_bytes"] == 0
        and meta["fetch_status"] == "fetch_failed"
    )
    text = await service.text_page(token.secret, attachment_id=1, limit=7)
    assert text["extracted_text"] == (None if owner_index == 0 else "private")
    assert text["extraction_complete"] is None
    assert text["next_offset"] == (None if owner_index == 0 else 7)
    first = await service.list_for_thread(token.secret, thread_id="thread", limit=1)
    assert [item["id"] for item in first["attachments"]] == [1] and first[
        "next_attachment_id"
    ] == 1
    last = await service.list_for_thread(
        token.secret, thread_id="thread", after_attachment_id=1, limit=1
    )
    assert (
        [item["id"] for item in last["attachments"]] == [2]
        and last["source_complete"]
        and not last["complete"]
    )
    assert last["next_attachment_id"] is None
    serialized = json.dumps([meta, text, first, last])
    assert "owner_id" not in serialized and owners[1 - owner_index] not in serialized
    assert not capacity.active
    held = capacity.acquire(owner)
    try:
        with pytest.raises(RuntimeError, match="^Attachment data is unavailable.$"):
            await service.describe(token.secret, attachment_id=1)
        assert capacity.active[owner] == 1
    finally:
        held.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "deadline"])
async def test_initial_authorization_thread_is_drained_within_service_lifecycle(
    tmp_path, monkeypatch, failure
):
    import threading

    service, caps, token, api = compose(tmp_path)
    started, release = threading.Event(), threading.Event()
    original = caps.authorize
    loop = asyncio.get_running_loop()
    clock = loop.time

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(caps, "authorize", blocked)
    task = asyncio.create_task(service.describe(token.secret, attachment_id=1))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        if failure == "cancel":
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
        else:
            monkeypatch.setattr(loop, "time", lambda: clock() + 31)
        await asyncio.sleep(0.01)
        assert not task.done() and not api.calls
        release.set()
        with pytest.raises(
            asyncio.CancelledError if failure == "cancel" else TimeoutError
        ):
            await task
    finally:
        monkeypatch.setattr(loop, "time", clock)
        release.set()
        await asyncio.gather(task, return_exceptions=True)
