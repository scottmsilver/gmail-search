"""Run-bound hybrid facts over one owner snapshot; no DDL or legacy loaders."""

import asyncio
from dataclasses import dataclass
import json
import math
from numbers import Real
import re
from types import MappingProxyType

import numpy as np

from .registry import AccessDenied
from .search_queries import FactVectorRow, Selection, _text

MAX_VECTORS = 200_000
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Same vendored Snowball stopwords as the legacy facts lexical path. Importing
# propositions would also import its database/provider/DDL implementation.
_STOP = frozenset(
    (
        "i me my myself we our ours ourselves you you're you've you'll you'd your yours "
        "yourself yourselves he him his himself she she's her hers herself it it's its "
        "itself they them their theirs themselves what which who whom this that that'll "
        "these those am is are was were be been being have has had having do does did "
        "doing a an the and but if or because as until while of at by for with about "
        "against between into through during before after above below to from up down in "
        "out on off over under again further then once here there when where why how all "
        "any both each few more most other some such no nor not only own same so than too "
        "very s t can will just don don't should should've now d ll m o re ve y ain aren "
        "aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven "
        "haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't "
        "shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't"
    ).split()
)


class FactsIncomplete(RuntimeError):
    def __init__(self):
        super().__init__("Fact search could not complete within its limits.")


@dataclass(frozen=True)
class FactsOwnerContext:
    owner_id: str
    identity: str

    def __post_init__(self):
        _text(self.owner_id)
        _text(self.identity, 1000)
        if not self.identity.strip():
            raise ValueError("Invalid trusted fact owner identity")


def _singularize(token):
    if len(token) <= 3 or token.endswith("ss"):
        return None
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "ses", "zes")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return None


def _query_terms(query):
    return [
        token
        for token in re.findall("[a-z0-9]+", query.lower())
        if len(token) > 1 and token not in _STOP
    ]


def _lexical_terms(query):
    terms = []
    for token in _query_terms(query):
        terms.append(token)
        singular = _singularize(token)
        if singular and singular != token and singular not in _STOP:
            terms.append(singular)
    if len(terms) > 128 or any(len(token) > 128 for token in terms):
        raise ValueError("Fact query exceeds lexical limits")
    return tuple(terms)


def _owner_terms(identity):
    identity = identity.lower()
    full = re.sub(r"\s*\(.*\)", "", identity).strip()
    match = re.search(r"\(([^@()]+)@", identity)
    return tuple(
        term for term in (full, match.group(1) if match else "") if len(term) >= 4
    )


async def _drain(task):
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def _thread(function, *args, **kwargs):
    return await _drain(
        asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    )


def _query_vector(vector, dimensions):
    if isinstance(vector, np.ndarray):
        valid = (
            vector.ndim == 1
            and vector.size == dimensions
            and vector.dtype.kind in "fiu"
        )
    else:
        valid = (
            isinstance(vector, (list, tuple))
            and len(vector) == dimensions
            and all(
                isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                for value in vector
            )
        )
    if not valid:
        raise RuntimeError("Invalid fact query vector")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.array(vector, dtype=np.float32, copy=True)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("Invalid fact query vector") from None
    norm = np.linalg.norm(values)
    if not np.isfinite(values).all() or not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid fact query vector")
    return values / (norm + 1e-8)


def _cosine(row, query, dimensions):
    if row.vector_status != "ok":
        if row.vector_status not in (
            "missing_vector",
            "model_mismatch",
            "invalid_vector",
        ):
            raise RuntimeError("Invalid fact vector status")
        return None, row.vector_status
    if not isinstance(row.embedding, bytes) or len(row.embedding) != dimensions * 4:
        return None, "invalid_vector"
    vector = np.frombuffer(row.embedding, dtype="<f4")
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
        return None, "invalid_vector"
    score = float((vector / (norm + 1e-8)) @ query)
    if not math.isfinite(score):
        return None, "invalid_vector"
    return max(-1.0, min(1.0, score)), None


def _selection(page, limit):
    if (
        not isinstance(page, Selection)
        or not isinstance(page.rows, (tuple, list))
        or len(page.rows) > limit
        or type(page.complete) is not bool
        or page.reason
        not in (None, "row_limit", "row_bytes", "page_bytes", "session_budget")
        or (page.complete and (page.reason is not None or page.next_cursor is not None))
        or (not page.complete and page.reason is None)
        or (
            page.next_cursor is not None
            and page.reason not in ("row_limit", "page_bytes")
        )
    ):
        raise RuntimeError("Invalid fact selection")


def _rank(scalars, lexical, exhaustive):
    semantic = sorted(
        (pid for pid, meta in scalars.items() if meta[0] is not None),
        key=lambda pid: (-scalars[pid][0], pid),
    )
    semantic_rank = {pid: rank for rank, pid in enumerate(semantic, 1)}
    lexical_rank = {pid: rank for rank, pid in enumerate(lexical, 1)}
    candidates = (
        ({pid for pid in semantic if scalars[pid][0] >= 0.5} | set(lexical))
        if exhaustive
        else set(semantic[:20]) | set(lexical[:20])
    )
    candidates.intersection_update(scalars)

    def score(pid):
        return (
            (1 / (60 + semantic_rank[pid]) if pid in semantic_rank else 0.0)
            + (1 / (60 + lexical_rank[pid]) if pid in lexical_rank else 0.0)
            + (0.02 if scalars[pid][1] else 0.0)
        )

    return sorted(candidates, key=lambda pid: (-score(pid), pid))


class RunFactsService:
    def __init__(self, capabilities, reader, embedder, *, owners):
        if len(owners) > 4096 or any(
            type(value) is not FactsOwnerContext or key != value.owner_id
            for key, value in owners.items()
        ):
            raise ValueError("Invalid trusted fact owner registry")
        self.capabilities = capabilities
        self.reader = reader
        self.embedder = embedder
        self.owners = MappingProxyType(dict(owners))

    async def _authorize(self, token):
        return await _thread(
            self.capabilities.authorize,
            token,
            audience="retrieval",
            operation="facts.find",
        )

    async def _hydrate(self, db, requested, reasons, check):
        rows = {}
        after = None
        while True:
            page = await db.fact_vectors_page(
                fact_ids=tuple(requested), after=after, limit=128
            )
            await check()
            _selection(page, 128)
            last = after
            for row in page.rows:
                if (
                    not isinstance(row, FactVectorRow)
                    or type(row.id) is not int
                    or row.id not in requested
                    or row.id in rows
                    or (last is not None and row.id <= last)
                ):
                    raise RuntimeError("Invalid fact hydration")
                last = row.id
                rows[row.id] = row
            if page.complete:
                if page.next_cursor is not None:
                    raise RuntimeError("Invalid fact hydration")
                break
            if page.next_cursor is None:
                reasons.add("hydration_" + (page.reason or "incomplete"))
                break
            if (
                not page.rows
                or type(page.next_cursor) is not int
                or page.next_cursor != last
            ):
                raise RuntimeError("Invalid fact hydration cursor")
            after = page.next_cursor
        if set(requested) - set(rows):
            reasons.add("unavailable_fact")
        return rows

    async def find_facts(self, token, *, query, exhaustive=True, k=200):
        deadline = asyncio.get_running_loop().time() + 30
        async with asyncio.timeout_at(deadline):
            lease = await self._authorize(token)
        _text(query, 1000)
        if (
            not query.strip()
            or type(exhaustive) is not bool
            or type(k) is not int
            or not 1 <= k <= 500
        ):
            raise ValueError("Invalid fact search options")
        terms = _lexical_terms(query)

        async def check():
            current = await self._authorize(token)
            if current.owner_id != lease.owner_id or current.run_id != lease.run_id:
                raise AccessDenied()
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Fact search deadline exceeded")
            return True

        async def watch():
            while True:
                await asyncio.sleep(0.05)
                await check()

        async def execute():
            async with asyncio.timeout_at(deadline):
                return await self._execute(
                    lease, query, terms, exhaustive, k, deadline, check
                )

        work = asyncio.create_task(execute())
        watcher = asyncio.create_task(watch())
        try:
            done, _ = await asyncio.wait(
                (work, watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                await watcher
            result = await work
        finally:
            work.cancel()
            watcher.cancel()
            await _drain(
                asyncio.ensure_future(
                    asyncio.gather(work, watcher, return_exceptions=True)
                )
            )
        # All owned cleanup must precede publication authorization: a cancelled
        # watcher can still be draining its authorization thread.
        await check()
        return result

    async def _execute(self, lease, query, terms, exhaustive, k, deadline, check):
        context = self.owners.get(lease.owner_id)
        profile = self.reader.profile
        if (
            context is None
            or self.embedder.model != profile.embedding_model
            or self.embedder.dimensions != profile.dimensions
        ):
            raise RuntimeError("Fact search profile is unavailable")
        vector = await self.embedder.embed(
            lease, query, deadline=deadline, check_active=check
        )
        query_vector = _query_vector(vector, profile.dimensions)
        await check()
        reasons = set()
        owner_terms = _owner_terms(context.identity)
        async with self.reader.session(
            lease.owner_id, deadline=deadline, check_active=check
        ) as db:
            total = await db.fact_count()
            if type(total) is not int or total < 0:
                raise RuntimeError("Invalid fact count")
            lexical = []
            if terms:
                page = await db.lexical_facts(terms, limit=k * 4)
                _selection(page, k * 4)
                if page.next_cursor is not None:
                    raise RuntimeError("Invalid fact selection")
                if any(
                    type(row.id) is not int
                    or not 0 < row.id < 2**63
                    or type(row.score) not in (int, float)
                    or not math.isfinite(row.score)
                    or row.score < 0
                    for row in page.rows
                ):
                    raise RuntimeError("Invalid fact lexical identity or score")
                if not page.complete:
                    reasons.add("lexical_" + (page.reason or "incomplete"))
                lexical = [row.id for row in page.rows]
                if len(set(lexical)) != len(lexical):
                    raise RuntimeError("Invalid fact lexical identity list")
            scalars = {}
            after = None
            while len(scalars) < MAX_VECTORS:
                page = await db.fact_vectors_page(
                    after=after, limit=min(128, MAX_VECTORS - len(scalars))
                )
                await check()
                _selection(page, min(128, MAX_VECTORS - len(scalars)))
                last = after
                for row in page.rows:
                    if (
                        not isinstance(row, FactVectorRow)
                        or type(row.id) is not int
                        or row.id <= 0
                        or row.id >= 2**63
                        or (last is not None and row.id <= last)
                    ):
                        raise RuntimeError("Invalid fact page")
                    last = row.id
                    score, reason = (
                        (None, "model_mismatch")
                        if row.model != profile.fact_model_tag
                        else _cosine(row, query_vector, profile.dimensions)
                    )
                    if reason:
                        reasons.add(reason)
                    scalars[row.id] = (
                        score,
                        any(term in row.text.lower() for term in owner_terms),
                    )
                if page.complete:
                    if page.next_cursor is not None:
                        raise RuntimeError("Invalid fact page")
                    break
                if page.next_cursor is None:
                    reasons.add("scan_" + (page.reason or "incomplete"))
                    break
                if (
                    not page.rows
                    or type(page.next_cursor) is not int
                    or page.next_cursor != last
                ):
                    raise RuntimeError("Invalid fact cursor")
                after = page.next_cursor
                await asyncio.sleep(0)
            else:
                reasons.add("vector_limit")
            if len(scalars) != total:
                reasons.add("incomplete_fact_scan")
            if any(pid not in scalars for pid in lexical):
                reasons.add("unavailable_fact")
            # Revalidate before rank positions are assigned: unavailable IDs
            # must not shift the surviving owner's reciprocal-rank scores.
            lexical = [pid for pid in lexical if pid in scalars]
            ordered = await _thread(_rank, scalars, lexical, exhaustive)
            facts = []
            seen = set()
            size = 0
            stopped = False
            for start in range(0, len(ordered), 128):
                requested = ordered[start : start + 128]
                rows = await self._hydrate(db, requested, reasons, check)
                for pid in requested:
                    row = rows.get(pid)
                    if row is None or row.text in seen:
                        continue
                    seen.add(row.text)
                    if len(facts) >= k:
                        reasons.add("result_limit")
                        stopped = True
                        break
                    entry = dict(
                        fact=row.text,
                        message_id=row.message_id,
                        thread_id=row.thread_id,
                        cosine=round(scalars[pid][0] or 0.0, 4),
                        bm25=pid in lexical,
                        owner=scalars[pid][1],
                    )
                    encoded = json.dumps(
                        entry, ensure_ascii=False, allow_nan=False
                    ).encode("utf-8")
                    if size + len(encoded) > MAX_RESPONSE_BYTES - 8192:
                        reasons.add("response_bytes")
                        stopped = True
                        break
                    facts.append(entry)
                    size += len(encoded)
                if stopped:
                    break
            if not facts and reasons:
                raise FactsIncomplete()
            result = dict(
                facts=facts,
                coverage=dict(
                    corpus_count=total,
                    examined=len(scalars),
                    selection_complete=not reasons,
                    mailbox_extraction_complete=False,
                    exhaustive_requested=exhaustive,
                    reasons=sorted(reasons),
                ),
            )
            if (
                len(
                    json.dumps(result, ensure_ascii=False, allow_nan=False).encode(
                        "utf-8"
                    )
                )
                > MAX_RESPONSE_BYTES
            ):
                raise FactsIncomplete()
            await check()
            return result
