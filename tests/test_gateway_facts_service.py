"""Synthetic facts orchestration; no production corpus or provider calls."""

import asyncio
from contextlib import asynccontextmanager
import importlib
import struct
from types import SimpleNamespace

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.gateway.search_queries import FactVectorRow, LexicalHit, Selection
from gmail_search.gateway.search_reader import SearchProfile


def fact(identifier, text, vector=(1.0, 0.0), *, model="fixture+v1", status="ok"):
    return FactVectorRow(
        identifier,
        "m" + str(identifier),
        "thread" + str(identifier),
        text,
        None if vector is None else struct.pack("<2f", *vector),
        model,
        status,
    )


class Queries:
    def __init__(self, rows):
        self.rows = rows
        self.lexical = []
        self.tokens = []
        self.pages = []

    async def fact_count(self):
        return len(self.rows)

    async def lexical_facts(self, tokens, *, limit):
        self.tokens.append(tokens)
        return Selection(
            tuple(self.lexical[:limit]),
            len(self.lexical) <= limit,
            "row_limit" if len(self.lexical) > limit else None,
        )

    async def fact_vectors_page(self, *, after=None, fact_ids=None, limit=128):
        assert 1 <= limit <= 128
        if fact_ids is not None:
            assert len(fact_ids) <= 128
        self.pages.append((after, fact_ids, limit))
        rows = [
            row
            for row in self.rows
            if (after is None or row.id > after)
            and (fact_ids is None or row.id in fact_ids)
        ]
        selected = tuple(rows[:limit])
        complete = len(rows) <= limit
        return Selection(
            selected,
            complete,
            None if complete else "row_limit",
            None if complete else selected[-1].id,
        )


class Reader:
    profile = SearchProfile("fixture", "fixture+v1", 2)

    def __init__(self, rows):
        self.queries = Queries(rows)
        self.open = 0
        self.owners = []

    @asynccontextmanager
    async def session(self, owner_id, *, deadline, check_active):
        self.owners.append(owner_id)
        self.open += 1
        try:
            await check_active()
            yield self.queries
        finally:
            self.open -= 1


class Embedder:
    model = "fixture"
    dimensions = 2

    def __init__(self, reader):
        self.reader = reader
        self.calls = []

    async def embed(self, lease, text, *, deadline, check_active):
        assert not self.reader.open
        await check_active()
        self.calls.append((lease.owner_id, text))
        return [1.0, 0.0]


def compose(tmp_path, rows):
    assert importlib.util.find_spec("gmail_search.gateway.facts_service") is not None
    module = importlib.import_module("gmail_search.gateway.facts_service")
    registry = Registry(tmp_path / "facts.sqlite", is_active=lambda _: True)
    capabilities = Capabilities(registry)
    run = registry.start_run("alice", "conversation", request_key="facts", writer=False)
    token = capabilities.issue(
        run.run_id, audience="retrieval", operations={"facts.find"}
    )
    reader = Reader(rows)
    embedder = Embedder(reader)
    service = module.RunFactsService(
        capabilities,
        reader,
        embedder,
        owners={
            "alice": module.FactsOwnerContext(
                "alice", "Alice Example (alice@example.test)"
            )
        },
    )
    return service, capabilities, token, reader, embedder


@pytest.mark.asyncio
async def test_hybrid_owner_boost_and_exact_text_dedup_preserve_contract(tmp_path):
    rows = [
        fact(1, "A neighbor owns a car"),
        fact(2, "Alice Example owns a car"),
        fact(3, "Alice Example owns a car"),
        fact(4, "Plate ABC123", (0.0, 1.0)),
    ]
    service, _, token, reader, embedder = compose(tmp_path, rows)
    reader.queries.lexical = [LexicalHit(4, "m4", 5.0)]
    result = await service.find_facts(token.secret, query="what cars do I own")
    assert result["facts"][0]["fact"] == "Alice Example owns a car"
    assert result["facts"][0]["owner"] is True
    assert len(result["facts"]) == 3
    plate = next(row for row in result["facts"] if row["fact"] == "Plate ABC123")
    assert plate == dict(
        fact="Plate ABC123",
        message_id="m4",
        thread_id="thread4",
        cosine=0.0,
        bm25=True,
        owner=False,
    )
    assert embedder.calls == [("alice", "what cars do I own")]
    assert reader.queries.tokens == [("cars", "car")]
    assert reader.owners == ["alice"] and not reader.open
    assert result["coverage"]["corpus_count"] == 4
    assert result["coverage"]["selection_complete"] is True
    assert result["coverage"]["mailbox_extraction_complete"] is False


@pytest.mark.asyncio
async def test_semantic_last_page_beyond_8000_has_no_lexical_dependency(tmp_path):
    rows = [fact(i, "unrelated " + str(i), (0.0, 1.0)) for i in range(1, 8002)]
    rows[-1] = fact(8001, "The desired semantic fact")
    service, _, token, reader, _ = compose(tmp_path, rows)
    result = await service.find_facts(token.secret, query="to be or not to be")
    assert [row["fact"] for row in result["facts"]] == ["The desired semantic fact"]
    assert result["coverage"]["examined"] == 8001
    assert result["coverage"]["selection_complete"] is True
    assert not reader.queries.tokens
    assert max(page[2] for page in reader.queries.pages) <= 128


@pytest.mark.asyncio
async def test_stable_ties_and_result_cap_are_explicit(tmp_path):
    service, _, token, _, _ = compose(
        tmp_path, [fact(i, "fact " + str(i)) for i in range(1, 5)]
    )
    result = await service.find_facts(token.secret, query="query", k=2)
    assert [row["message_id"] for row in result["facts"]] == ["m1", "m2"]
    assert "result_limit" in result["coverage"]["reasons"]
    assert result["coverage"]["selection_complete"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"query": ""},
        {"query": "x", "exhaustive": 1},
        {"query": "x", "k": True},
        {"query": "x", "k": 501},
        {"query": "x", "owner_id": "bob"},
    ],
)
async def test_fixed_options_rejected_before_reader_or_provider(tmp_path, options):
    service, _, token, reader, embedder = compose(tmp_path, [])
    with pytest.raises((ValueError, TypeError)):
        await service.find_facts(token.secret, **options)
    assert not reader.owners and not embedder.calls


@pytest.mark.asyncio
async def test_revoked_capability_never_starts_work(tmp_path):
    service, caps, token, reader, embedder = compose(tmp_path, [])
    caps.revoke(token.secret)
    with pytest.raises(AccessDenied):
        await service.find_facts(token.secret, query="query")
    assert not reader.owners and not embedder.calls


@pytest.mark.asyncio
async def test_unhydrated_lexical_ids_never_change_own_rrf_ranks(tmp_path):
    rows = [fact(i, "item " + str(i), (1.0, i / 100)) for i in range(1, 101)]
    service, _, token, reader, _ = compose(tmp_path, rows)
    reader.queries.lexical = [LexicalHit(100, "m100", 1.0)]
    baseline = await service.find_facts(token.secret, query="item", k=500)
    reader.queries.lexical = [
        LexicalHit(i, "missing", 100.0) for i in range(1000, 1700)
    ] + [LexicalHit(100, "m100", 1.0)]
    polluted = await service.find_facts(token.secret, query="item", k=500)
    assert polluted["facts"] == baseline["facts"]
    assert "unavailable_fact" in polluted["coverage"]["reasons"]


@pytest.mark.asyncio
async def test_wrong_model_cannot_score_even_when_row_status_claims_ok(tmp_path):
    rows = [fact(1, "Wrong model", model="other"), fact(2, "Valid model")]
    service, _, token, _, _ = compose(tmp_path, rows)
    result = await service.find_facts(token.secret, query="model")
    assert [row["fact"] for row in result["facts"]] == ["Valid model"]
    assert "model_mismatch" in result["coverage"]["reasons"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [(1, 1), (0,), (-1,), (True,)])
async def test_invalid_lexical_identity_list_fails_closed(tmp_path, ids):
    service, _, token, reader, _ = compose(tmp_path, [fact(1, "one")])
    # Supply malformed reader output directly: LexicalHit itself now rejects
    # nonnumeric keys, while this regression exercises the service's boundary.
    reader.queries.lexical = [SimpleNamespace(id=i, message_id="m1", score=1.0) for i in ids]
    with pytest.raises(RuntimeError, match="Invalid fact lexical"):
        await service.find_facts(token.secret, query="one")


@pytest.mark.asyncio
async def test_nonexhaustive_selection_keeps_top20_and_distinct_values(tmp_path):
    service, _, token, _, _ = compose(
        tmp_path, [fact(i, "Plate ABC" + str(i)) for i in range(1, 41)]
    )
    result = await service.find_facts(token.secret, query="plates", exhaustive=False)
    assert len(result["facts"]) == 20
    assert result["facts"][0]["fact"] == "Plate ABC1"
    assert result["facts"][-1]["fact"] == "Plate ABC20"
    assert result["coverage"]["exhaustive_requested"] is False


@pytest.mark.asyncio
async def test_missing_and_invalid_vectors_preserve_lexical_facts_with_coverage(
    tmp_path,
):
    rows = [
        fact(1, "Missing", None, status="missing_vector"),
        fact(2, "Zero", (0.0, 0.0)),
        fact(3, "Nonfinite", (float("nan"), 0.0)),
        fact(4, "Valid"),
    ]
    service, _, token, reader, _ = compose(tmp_path, rows)
    reader.queries.lexical = [LexicalHit(1, "m1", 1.0)]
    result = await service.find_facts(token.secret, query="query")
    assert {row["fact"] for row in result["facts"]} == {"Missing", "Valid"}
    assert {"missing_vector", "invalid_vector"} <= set(result["coverage"]["reasons"])


@pytest.mark.asyncio
async def test_incomplete_empty_scan_is_fixed_error_not_empty_success(tmp_path):
    service, _, token, reader, _ = compose(tmp_path, [fact(1, "unseen")])

    async def blocked(**kwargs):
        return Selection((), False, "session_budget")

    reader.queries.fact_vectors_page = blocked
    with pytest.raises(
        RuntimeError, match="Fact search could not complete within its limits"
    ):
        await service.find_facts(token.secret, query="query")
    assert not reader.open


@pytest.mark.asyncio
async def test_vector_cap_returns_partial_positive_results(tmp_path, monkeypatch):
    service, _, token, _, _ = compose(
        tmp_path, [fact(i, "fact " + str(i)) for i in range(1, 5)]
    )
    module = importlib.import_module("gmail_search.gateway.facts_service")
    monkeypatch.setattr(module, "MAX_VECTORS", 2)
    result = await service.find_facts(token.secret, query="query")
    assert len(result["facts"]) == 2
    assert result["coverage"]["examined"] == 2
    assert "vector_limit" in result["coverage"]["reasons"]


@pytest.mark.asyncio
async def test_nonadvancing_cursor_refuses_partial_data(tmp_path):
    service, _, token, reader, _ = compose(tmp_path, [fact(1, "fact")])

    async def invalid(**kwargs):
        return Selection((fact(1, "fact"),), False, "row_limit", 0)

    reader.queries.fact_vectors_page = invalid
    with pytest.raises(RuntimeError, match="Invalid fact cursor"):
        await service.find_facts(token.secret, query="query")


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_provider_cleanup(tmp_path):
    service, _, token, reader, embedder = compose(tmp_path, [])
    entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()

    embedder.embed = blocked
    task = asyncio.create_task(service.find_facts(token.secret, query="query"))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    await asyncio.wait_for(closing.wait(), 1)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done() and not reader.open
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_revocation_during_scan_prevents_publication(tmp_path):
    service, caps, token, reader, _ = compose(tmp_path, [fact(1, "private fact")])
    original = reader.queries.fact_vectors_page

    async def revoked(**kwargs):
        page = await original(**kwargs)
        caps.revoke(token.secret)
        return page

    reader.queries.fact_vectors_page = revoked
    with pytest.raises(AccessDenied):
        await service.find_facts(token.secret, query="query")
    assert not reader.open


@pytest.mark.asyncio
async def test_total_output_limit_includes_coverage_and_preserves_complete_facts(
    tmp_path,
):
    import json

    service, _, token, _, _ = compose(
        tmp_path, [fact(i, "x" * 9000 + str(i)) for i in range(1, 501)]
    )
    result = await service.find_facts(token.secret, query="query", k=500)
    assert 0 < len(result["facts"]) < 500
    assert all(len(row["fact"]) >= 9001 for row in result["facts"])
    assert "response_bytes" in result["coverage"]["reasons"]
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 4 * 1024 * 1024


@pytest.mark.asyncio
async def test_nested_provider_vector_is_rejected_before_numpy_conversion(
    tmp_path, monkeypatch
):
    service, _, token, reader, embedder = compose(tmp_path, [])
    module = importlib.import_module("gmail_search.gateway.facts_service")

    async def invalid(*args, **kwargs):
        return [[1.0, 0.0], [1.0, 0.0]]

    embedder.embed = invalid

    def forbidden(*args, **kwargs):
        pytest.fail("Nested vector reached numpy conversion")

    monkeypatch.setattr(module.np, "array", forbidden)
    monkeypatch.setattr(module.np, "asarray", forbidden)
    with pytest.raises(RuntimeError, match="Invalid fact query vector"):
        await service.find_facts(token.secret, query="query")
    assert not reader.owners


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_snapshot_until_ranking_thread_finishes(
    tmp_path, monkeypatch
):
    import threading

    service, _, token, reader, _ = compose(tmp_path, [fact(1, "one")])
    module = importlib.import_module("gmail_search.gateway.facts_service")
    original = module._rank
    started, release = threading.Event(), threading.Event()

    def delayed(*args):
        started.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(module, "_rank", delayed)
    task = asyncio.create_task(service.find_facts(token.secret, query="query"))
    assert await asyncio.to_thread(started.wait, 1)
    try:
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done() and reader.open == 1
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not reader.open


@pytest.mark.asyncio
async def test_rehydration_uses_continuation_without_losing_ranked_facts(tmp_path):
    service, _, token, reader, _ = compose(
        tmp_path, [fact(i, "fact " + str(i)) for i in range(1, 5)]
    )
    original = reader.queries.fact_vectors_page

    async def tiny(**kwargs):
        if kwargs.get("fact_ids") is not None:
            kwargs["limit"] = 1
        return await original(**kwargs)

    reader.queries.fact_vectors_page = tiny
    result = await service.find_facts(token.secret, query="query")
    assert len(result["facts"]) == 4
    assert result["coverage"]["selection_complete"] is True


from test_gateway_search_reader import search_database as search_database, provisioned


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_index", [0, 1])
async def test_real_reader_isolates_colliding_fact_ids_and_foreign_anchors(
    search_database, tmp_path, owner_index
):
    from gmail_search.gateway.data_admission import DataAdmission
    from gmail_search.gateway.search_reader import SearchReader, SearchRegistry

    module = importlib.import_module("gmail_search.gateway.facts_service")
    db = search_database
    owner = db.owners[owner_index]
    word, foreign = (
        ("needle", "foreignonly") if owner_index == 0 else ("foreignonly", "needle")
    )
    credential = provisioned(db, owner)
    capacity = DataAdmission(global_concurrency=1, owner_concurrency=1)
    reader = SearchReader(
        SearchRegistry({owner: credential}, is_active=lambda _: True),
        profile=SearchProfile("fixture", "fixture+v1", 2),
        admission=capacity,
    )
    registry = Registry(tmp_path / "real-facts.sqlite", is_active=lambda _: True)
    capabilities = Capabilities(registry)
    run = registry.start_run(owner, "conversation", request_key="facts", writer=False)
    token = capabilities.issue(
        run.run_id, audience="retrieval", operations={"facts.find"}
    )

    class RealFixtureEmbedder:
        model = "fixture"
        dimensions = 2

        async def embed(self, lease, text, *, deadline, check_active):
            assert lease.owner_id == owner and not capacity.active
            await check_active()
            return [1.0, 0.0] if text == word else [0.0, 1.0]

    service = module.RunFactsService(
        capabilities,
        reader,
        RealFixtureEmbedder(),
        owners={owner: module.FactsOwnerContext(owner, owner + " (user@example.test)")},
    )
    result = await service.find_facts(token.secret, query=word)
    assert result["facts"] == [
        dict(
            fact=word,
            message_id="same",
            thread_id="thread",
            cosine=1.0,
            bm25=True,
            owner=False,
        )
    ]
    assert result["coverage"]["corpus_count"] == 1
    assert result["coverage"]["selection_complete"] is True
    assert not capacity.active
    assert (await service.find_facts(token.secret, query=foreign))["facts"] == []
    assert not capacity.active


@pytest.mark.asyncio
async def test_revocation_during_final_watcher_drain_prevents_publication(
    tmp_path, monkeypatch
):
    import threading

    service, caps, token, _, _ = compose(tmp_path, [fact(1, "private fact")])
    module = importlib.import_module("gmail_search.gateway.facts_service")
    loop = asyncio.get_running_loop()
    started, closing = asyncio.Event(), asyncio.Event()
    gate = threading.Event()
    original_auth, original_execute, original_drain = (
        service._authorize,
        service._execute,
        module._drain,
    )

    def delayed_auth():
        result = caps.authorize(
            token.secret, audience="retrieval", operation="facts.find"
        )
        loop.call_soon_threadsafe(started.set)
        assert gate.wait(3)
        return result

    async def authorize(value):
        if asyncio.current_task().get_coro().__qualname__.endswith(".watch"):
            return await module._thread(delayed_auth)
        return await original_auth(value)

    async def execute(*args):
        await started.wait()
        return await original_execute(*args)

    async def drain(task):
        if type(task).__name__ == "_GatheringFuture":
            closing.set()
        return await original_drain(task)

    monkeypatch.setattr(service, "_authorize", authorize)
    monkeypatch.setattr(service, "_execute", execute)
    monkeypatch.setattr(module, "_drain", drain)
    task = asyncio.create_task(service.find_facts(token.secret, query="query"))
    try:
        await asyncio.wait_for(closing.wait(), 2)
        caps.revoke(token.secret)
        gate.set()
        with pytest.raises(AccessDenied):
            await task
    finally:
        gate.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["lexical", "scan", "hydration"])
async def test_reader_page_diagnostics_are_closed_and_never_published(
    tmp_path, operation
):
    service, _, token, reader, _ = compose(tmp_path, [fact(1, "one")])
    original = reader.queries.fact_vectors_page

    async def invalid_lexical(*args, **kwargs):
        return Selection((LexicalHit(1, "m1", 1.0),), False, "private diagnostic")

    async def invalid_vector(**kwargs):
        if operation == "scan" or kwargs.get("fact_ids") is not None:
            return Selection((fact(1, "one"),), False, "private diagnostic")
        return await original(**kwargs)

    if operation == "lexical":
        reader.queries.lexical_facts = invalid_lexical
    else:
        reader.queries.fact_vectors_page = invalid_vector
    with pytest.raises(RuntimeError, match="^Invalid fact selection$"):
        await service.find_facts(token.secret, query="query")


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[], [fact(1, "unrelated", (0.0, 1.0))]])
async def test_complete_empty_selection_is_distinct_from_incomplete_search(
    tmp_path, rows
):
    service, _, token, _, _ = compose(tmp_path, rows)
    result = await service.find_facts(token.secret, query="the and a")
    assert result["facts"] == []
    assert result["coverage"]["selection_complete"] is True
    assert result["coverage"]["corpus_count"] == len(rows)
    assert result["coverage"]["mailbox_extraction_complete"] is False


@pytest.mark.asyncio
async def test_hydration_budget_exhaustion_cannot_be_successful_empty_search(tmp_path):
    service, _, token, reader, _ = compose(tmp_path, [fact(1, "positive")])
    original = reader.queries.fact_vectors_page

    async def exhausted(**kwargs):
        if kwargs.get("fact_ids") is not None:
            return Selection((), False, "session_budget")
        return await original(**kwargs)

    reader.queries.fact_vectors_page = exhausted
    with pytest.raises(
        RuntimeError, match="^Fact search could not complete within its limits.$"
    ):
        await service.find_facts(token.secret, query="query")
    assert not reader.open
