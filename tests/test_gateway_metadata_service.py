"""Synthetic structured metadata queries; no provider or production resources."""

import asyncio
import importlib
import json

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.database import QueryResult
from gmail_search.gateway.registry import AccessDenied, Registry

COLUMNS = (
    "owner_id",
    "thread_id",
    "summary_id",
    "subject",
    "participants",
    "message_count",
    "date_first",
    "date_last",
    "snippet",
)


def row(owner="alice", thread="thread", **changes):
    values = dict(
        zip(
            COLUMNS,
            (
                owner,
                thread,
                thread,
                "subject",
                '["sender@example.test"]',
                2,
                "2026-01-01",
                "2026-09-15",
                "snippet",
            ),
        )
    )
    values.update(changes)
    return tuple(values[key] for key in COLUMNS)


class Gateway:
    def __init__(self, rows=(), complete=True):
        self.result = QueryResult(COLUMNS, tuple(rows), complete)
        self.calls = []

    async def query(self, owner, query):
        self.calls.append((owner, query))
        return self.result


def compose(tmp_path, gateway=None, owner="alice"):
    module = importlib.import_module("gmail_search.gateway.metadata_service")
    registry = Registry(tmp_path / "metadata.sqlite", is_active=lambda _: True)
    caps = Capabilities(registry)
    run = registry.start_run(
        owner, "conversation", request_key="metadata", writer=False
    )
    token = caps.issue(run.run_id, audience="retrieval", operations={"query.emails"})
    api = gateway or Gateway([row(owner)])
    return module.RunMetadataService(caps, api), caps, token, api


@pytest.mark.asyncio
async def test_fixed_query_compiles_without_model_sql_and_preserves_output(tmp_path):
    from gmail_search.gateway.analytics import compile_query

    service, _, token, api = compose(tmp_path)
    result = await service.query_emails(
        token.secret,
        sender="o'brien%",
        subject_contains="a_b",
        label="INBOX",
        has_attachment=True,
    )
    compiled = compile_query(api.calls[0][1])
    assert api.calls[0][0] == "alice"
    assert "%o'brien%%" in compiled.params and "%a_b%" in compiled.params
    assert "EXISTS" in compiled.sql and "row_number" in compiled.sql
    assert result["results"][0] == dict(
        thread_id="thread",
        subject="subject",
        participants=["sender@example.test"],
        message_count=2,
        date_first="2026-01-01",
        date_last="2026-09-15",
        snippet="snippet",
        cite_ref="thread",
    )
    assert result["coverage"]["selection_complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"limit": True},
        {"limit": 101},
        {"sender": None},
        {"date_from": "2026-02-30"},
        {"date_from": "2026-09-15", "date_to": "2026-01-01"},
        {"has_attachment": 1},
        {"order_by": "random"},
        {"owner_id": "bob"},
    ],
)
async def test_invalid_options_never_query(tmp_path, options):
    service, _, token, api = compose(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        await service.query_emails(token.secret, **options)
    assert not api.calls


@pytest.mark.asyncio
async def test_limit_plus_one_and_incomplete_empty_are_explicit(tmp_path):
    service, _, token, api = compose(
        tmp_path, Gateway([row(thread="a"), row(thread="b")])
    )
    result = await service.query_emails(token.secret, limit=1)
    assert len(result["results"]) == 1
    assert result["coverage"]["reasons"] == ["result_limit"]
    api.result = QueryResult(COLUMNS, (), False)
    with pytest.raises(
        RuntimeError, match="Metadata query could not complete within its limits"
    ):
        await service.query_emails(token.secret)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        row(owner="bob"),
        row(participants="[1]"),
        row(message_count=True),
        row(summary_id="other"),
    ],
)
async def test_invalid_reader_rows_refuse_publication(tmp_path, bad):
    service, _, token, _ = compose(tmp_path, Gateway([bad]))
    with pytest.raises(RuntimeError, match="Invalid metadata result"):
        await service.query_emails(token.secret)


@pytest.mark.asyncio
async def test_missing_summary_and_complete_empty_selection_are_distinct(tmp_path):
    service, _, token, api = compose(tmp_path, Gateway([row(summary_id=None)]))
    with pytest.raises(RuntimeError, match="Metadata query could not complete"):
        await service.query_emails(token.secret)
    api.result = QueryResult(COLUMNS, (), True)
    assert (await service.query_emails(token.secret))["coverage"]["selection_complete"]


@pytest.mark.asyncio
async def test_null_body_becomes_empty_snippet_like_legacy(tmp_path):
    service, _, token, _ = compose(tmp_path, Gateway([row(snippet=None)]))
    assert (await service.query_emails(token.secret))["results"][0]["snippet"] == ""


from test_gateway_database_integration import database as database, reader_dsn


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_index", [0, 1])
async def test_real_owner_collisions_matching_order_latest_snippet_and_attachment_semantics(
    database, tmp_path, owner_index
):
    import psycopg
    from gmail_search.gateway.database import (
        QueryGateway,
        ReaderCredential,
        ReaderRegistry,
    )
    from gmail_search.gateway.data_admission import DataAdmission

    dsn, owners = database
    with psycopg.connect(dsn, autocommit=True) as conn:
        for owner in owners:
            for identifier, tid, sender, day, body in (
                ("a1", "a", "match", "2026-01-10", "old"),
                ("a2", "a", "other", "2026-09-15", owner + " latest a " * 100),
                ("b1", "b", "match", "2026-02-10", "old"),
                ("b2", "b", "other", "2026-03-01", owner + " latest b"),
            ):
                conn.execute(
                    "INSERT INTO messages(id,thread_id,from_addr,subject,date,labels,body_text,user_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        identifier,
                        tid,
                        sender + "@" + owner,
                        "Invoice 100%",
                        day + "T12:00:00+00:00",
                        '["INBOX"]',
                        body,
                        owner,
                    ),
                )
            for tid, last in (("a", "2026-09-15"), ("b", "2026-03-01")):
                conn.execute(
                    "INSERT INTO thread_summary(thread_id,subject,participants,message_count,date_first,date_last,user_id) VALUES(%s,%s,%s,2,%s,%s,%s)",
                    (
                        tid,
                        owner + " summary",
                        json.dumps([owner]),
                        "2026-01-01",
                        last,
                        owner,
                    ),
                )
            conn.execute(
                "INSERT INTO attachments(id,message_id,filename,user_id) VALUES(1,%s,%s,%s)",
                ("a1", owner, owner),
            )
        conn.execute(
            "INSERT INTO attachments(id,message_id,filename,user_id) VALUES(2,%s,%s,%s)",
            ("b1", "foreign", owners[1]),
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
    result = await service.query_emails(
        token.secret, sender="match", label="INBOX", subject_contains="Invoice"
    )
    assert [r["thread_id"] for r in result["results"]] == ["b", "a"]
    assert all(
        r["subject"] == owner + " summary" and r["participants"] == [owner]
        for r in result["results"]
    )
    assert result["results"][1]["snippet"] == (owner + " latest a " * 100)[:500]
    assert all(r["cite_ref"] == r["thread_id"] for r in result["results"])
    assert [
        r["thread_id"]
        for r in (
            await service.query_emails(
                token.secret, sender="match", order_by="date_asc"
            )
        )["results"]
    ] == ["a", "b"]
    limited = await service.query_emails(token.secret, sender="match", limit=1)
    assert limited["results"][0]["thread_id"] == "b" and limited["coverage"][
        "reasons"
    ] == ["result_limit"]
    absent = await service.query_emails(
        token.secret, sender="match", has_attachment=False
    )
    assert [r["thread_id"] for r in absent["results"]] == (
        ["b"] if owner_index == 0 else []
    )
    assert (
        len(
            (
                await service.query_emails(
                    token.secret, label="INBOX", has_attachment=False
                )
            )["results"]
        )
        == 2
    )
    assert (await service.query_emails(token.secret, sender=owners[1 - owner_index]))[
        "results"
    ] == []
    assert (await service.query_emails(token.secret, sender="MATCH"))["results"] == []
    assert (
        len(
            (
                await service.query_emails(
                    token.secret,
                    sender="match",
                    date_from="2026-02-10",
                    date_to="2026-02-10",
                )
            )["results"]
        )
        == 1
    )
    assert not capacity.active
    held = capacity.acquire(owner)
    try:
        with pytest.raises(RuntimeError, match="Data query capacity reached"):
            await service.query_emails(token.secret, sender="match")
        assert capacity.active[owner] == 1
    finally:
        held.release()
    assert not capacity.active


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["revoked", "audience", "operation"])
async def test_capability_failure_never_queries(tmp_path, kind):
    service, caps, token, api = compose(tmp_path)
    if kind == "revoked":
        caps.revoke(token.secret)
    else:
        lease = caps.authorize(
            token.secret, audience="retrieval", operation="query.emails"
        )
        token = caps.issue(
            lease.run_id,
            audience="sql" if kind == "audience" else "retrieval",
            operations={"query.emails"} if kind == "audience" else {"thread.get"},
        )
    with pytest.raises(AccessDenied):
        await service.query_emails(token.secret)
    assert not api.calls


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_query_cleanup_owned(tmp_path):
    service, _, token, api = compose(tmp_path)
    entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()

    api.query = blocked
    task = asyncio.create_task(service.query_emails(token.secret))
    try:
        await asyncio.wait_for(entered.wait(), 1)
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
async def test_revocation_cancels_inflight_query(tmp_path):
    service, caps, token, api = compose(tmp_path)
    entered, closed = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    api.query = blocked
    task = asyncio.create_task(service.query_emails(token.secret))
    await asyncio.wait_for(entered.wait(), 1)
    caps.revoke(token.secret)
    with pytest.raises(AccessDenied):
        await asyncio.wait_for(task, 1)
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["revoke", "deadline"])
async def test_final_authorization_follows_watcher_thread_drain(
    tmp_path, monkeypatch, failure
):
    import threading

    module = importlib.import_module("gmail_search.gateway.metadata_service")
    service, caps, token, api = compose(tmp_path)
    loop = asyncio.get_running_loop()
    started, closing = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    original_auth, original_query, original_drain, original_time = (
        service.authorize,
        api.query,
        module._drain,
        loop.time,
    )

    def delayed():
        lease = caps.authorize(
            token.secret, audience="retrieval", operation="query.emails"
        )
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3)
        return lease

    async def auth(value):
        if asyncio.current_task().get_coro().__qualname__.endswith(".watch"):
            return await module._drain(asyncio.to_thread(delayed))
        return await original_auth(value)

    async def query(*args):
        await started.wait()
        return await original_query(*args)

    async def drain(awaitable):
        if type(awaitable).__name__ == "_GatheringFuture":
            closing.set()
        return await original_drain(awaitable)

    monkeypatch.setattr(service, "authorize", auth)
    monkeypatch.setattr(api, "query", query)
    monkeypatch.setattr(module, "_drain", drain)
    task = asyncio.create_task(service.query_emails(token.secret))
    try:
        await asyncio.wait_for(closing.wait(), 2)
        if failure == "revoke":
            caps.revoke(token.secret)
        else:
            monkeypatch.setattr(loop, "time", lambda: original_time() + 31)
        release.set()
        with pytest.raises(AccessDenied if failure == "revoke" else TimeoutError):
            await task
    finally:
        monkeypatch.setattr(loop, "time", original_time)
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_total_output_budget_preserves_atomic_rows_and_coverage(
    tmp_path, monkeypatch
):
    module = importlib.import_module("gmail_search.gateway.metadata_service")
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 10000)
    service, _, token, _ = compose(
        tmp_path,
        Gateway(
            [row(thread="a", subject="x" * 4000), row(thread="b", subject="y" * 4000)]
        ),
    )
    result = await service.query_emails(token.secret)
    assert len(result["results"]) == 1 and result["results"][0]["subject"] == "x" * 4000
    assert result["coverage"]["reasons"] == ["response_bytes"]
    assert len(json.dumps(result).encode()) <= 10000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        QueryResult(COLUMNS, (), 1),
        QueryResult(("private diagnostic",), (), True),
        QueryResult(COLUMNS, (row(), row()), True),
        QueryResult(COLUMNS, (row(participants="[NaN]"),), True),
    ],
)
async def test_malformed_result_metadata_has_fixed_error(tmp_path, result):
    service, _, token, api = compose(tmp_path)
    api.result = result
    with pytest.raises(RuntimeError, match="^Invalid metadata result$"):
        await service.query_emails(token.secret)
