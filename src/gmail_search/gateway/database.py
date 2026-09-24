"""Immutable owner-specific readers; no privileged login or SET ROLE fallback.

This module is internal to the trusted gateway. Neither credentials nor an
owner selector belong in the agent-facing request schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import asyncio
import hashlib
import json
from types import MappingProxyType
from typing import Callable, Mapping

from psycopg.conninfo import conninfo_to_dict
import psycopg

from .bounded_fetch import bounded_rows
from .data_admission import DataAdmission


def reader_role(owner_id: str) -> str:
    if not isinstance(owner_id, str) or not owner_id or '\x00' in owner_id or len(owner_id) > 2048:
        raise ValueError('A valid stable owner ID is required')
    return 'gms_reader_' + hashlib.sha256(owner_id.encode()).hexdigest()[:40]


@dataclass(frozen=True)
class ReaderCredential:
    owner_id: str
    dsn: str = field(repr=False)

    def __post_init__(self):
        try:
            parsed = conninfo_to_dict(self.dsn)
        except Exception:
            raise ValueError('Invalid reader connection configuration') from None
        if parsed.get('user') != reader_role(self.owner_id):
            raise ValueError('Reader login does not match its owner')
        if parsed.get('service') or parsed.get('options'):
            raise ValueError('Reader connections cannot override server settings')


class ReaderRegistry:
    def __init__(self, credentials: Mapping[str, ReaderCredential], *, is_active: Callable[[str], bool]):
        if any(key != credential.owner_id for key, credential in credentials.items()):
            raise ValueError('Reader registry owner mismatch')
        self._credentials = MappingProxyType(dict(credentials))
        self._is_active = is_active

    def credential(self, owner_id: str) -> ReaderCredential:
        credential = self._credentials.get(owner_id)
        if credential is None or not self._is_active(owner_id):
            raise PermissionError('Mailbox access is unavailable')
        return credential


@dataclass(frozen=True)
class QueryLimits:
    max_rows: int = 1000
    max_bytes: int = 2_000_000
    global_concurrency: int = 8
    owner_concurrency: int = 2
    deadline_seconds: float = 10
    lock_timeout_ms: int = 1000

    def __post_init__(self):
        for name in ('max_rows', 'max_bytes', 'global_concurrency', 'owner_concurrency', 'lock_timeout_ms'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Query limits must be positive')
        if (type(self.deadline_seconds) not in (int, float) or not 0 < self.deadline_seconds <= 30
                or self.max_rows > 10_000 or self.max_bytes > 10_000_000
                or not self.owner_concurrency <= self.global_concurrency <= 32
                or self.lock_timeout_ms > 30_000):
            raise ValueError('Query limits exceed the analytical service maximum')


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple, ...]
    complete: bool


class QueryGateway:
    """Bounded SELECT execution, internal to one gateway event loop/process.

    Per-process counters complement the database role's hard connection limit.
    A multi-process gateway must add shared admission before increasing workers.
    Connections are always discarded, including after cancellation or errors.
    """

    def __init__(self, registry: ReaderRegistry, *, limits: QueryLimits | None = None, admission: DataAdmission | None = None):
        self.registry = registry
        self.limits = limits or QueryLimits()
        self.admission = admission if admission is not None else DataAdmission(
            global_concurrency=self.limits.global_concurrency,
            owner_concurrency=self.limits.owner_concurrency)
        if (self.admission.global_concurrency > self.limits.global_concurrency
                or self.admission.owner_concurrency > self.limits.owner_concurrency):
            raise ValueError('Shared admission exceeds analytical query profile')

    @property
    def active_queries(self):
        return self.admission.active

    async def query(self, owner_id: str, query: str) -> QueryResult:
        from .analytics import compile_query

        credential = self.registry.credential(owner_id)
        compiled = compile_query(query)
        admission_lease = self.admission.acquire(owner_id)
        tasks = []
        try:
            async with asyncio.timeout(self.limits.deadline_seconds):
                run = asyncio.create_task(self._execute(credential, compiled))
                watch = asyncio.create_task(self._watch_owner(owner_id))
                tasks = [run, watch]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if watch in done:
                    await watch
                result = await run
                self.registry.credential(owner_id)
                return result
        finally:
            for task in tasks:
                task.cancel()
            interrupted = False
            if tasks:
                draining = asyncio.gather(*tasks, return_exceptions=True)
                while not draining.done():
                    try:
                        await asyncio.shield(draining)
                    except asyncio.CancelledError:
                        interrupted = True
                draining.result()
            admission_lease.release()
            if interrupted:
                raise asyncio.CancelledError

    async def _watch_owner(self, owner_id):
        while True:
            self.registry.credential(owner_id)
            await asyncio.sleep(.1)

    async def _execute(self, credential, compiled):
        try:
            async with await psycopg.AsyncConnection.connect(
                credential.dsn, autocommit=True, connect_timeout=3,
                application_name='gms-analytical-gateway',
            ) as conn:
                cursor = await conn.execute("""SELECT session_user, current_user,
                    rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls,
                    EXISTS(SELECT 1 FROM pg_auth_members WHERE member=r.oid OR roleid=r.oid),
                    shobj_description(r.oid,'pg_authid')
                    FROM pg_roles r WHERE rolname=current_user"""
                )
                role = await cursor.fetchone()
                expected = reader_role(credential.owner_id)
                if not role or role[:2] != (expected, expected) or role[2] or role[3] or role[4] != 'gmail-search analytical reader v1 owner=' + credential.owner_id:
                    raise PermissionError('Database reader identity is not qualified')
                async with conn.transaction():
                    await conn.execute('SET TRANSACTION READ ONLY')
                    for setting, value in (
                        ('search_path', 'pg_catalog'), ('row_security', 'on'),
                        ('statement_timeout', str(max(1, int(self.limits.deadline_seconds * 1000)))),
                        ('lock_timeout', str(self.limits.lock_timeout_ms)),
                        ('work_mem', '4MB'), ('max_parallel_workers_per_gather', '0'),
                        # Results are always read to the end: plan for all rows, not a
                        # fast first row (a thread's attachment list took ~1 s otherwise).
                        ('cursor_tuple_fraction', '1'),
                    ):
                        await conn.execute('SELECT pg_catalog.set_config(%s,%s,true)', (setting, value))
                    # temp_file_limit is an administrator-only setting pinned at provisioning.
                    cursor = await conn.execute("SELECT current_setting('temp_file_limit')")
                    if (await cursor.fetchone())[0] != '64MB':
                        raise PermissionError('Database reader resource limit is not qualified')
                    rows, size, complete = [], 0, True
                    async with conn.cursor(name='analytical_result') as cursor:
                        # Bound each row *before* PostgreSQL sends it. Fetching then
                        # measuring would already allocate arbitrary text in Python.
                        # The compiler also excludes text/array amplification routines.
                        bounded = ('SELECT CASE WHEN pg_catalog.octet_length(pg_catalog.row_to_json(gms_result)::pg_catalog.text) <= %s '
                                   'THEN pg_catalog.row_to_json(gms_result)::pg_catalog.text ELSE NULL END '
                                   'FROM (' + compiled.sql + ') AS gms_result')
                        await cursor.execute(bounded, (self.limits.max_bytes, *compiled.params))

                        async def still_active():
                            self.registry.credential(credential.owner_id)
                        # Liveness is checked per batch (and by _watch_owner), never per row.
                        async with bounded_rows(cursor, row_byte_cap=self.limits.max_bytes,
                                                between_batches=still_active) as results:
                            async for (payload,) in results:
                                if payload is None or len(rows) >= self.limits.max_rows:
                                    complete = False
                                    break
                                size += len(payload.encode())
                                if size > self.limits.max_bytes:
                                    complete = False
                                    break
                                values = json.loads(payload, parse_float=Decimal)
                                rows.append(tuple(values[column] for column in compiled.columns))
                    return QueryResult(compiled.columns, tuple(rows), complete)
        except psycopg.errors.QueryCanceled:
            raise TimeoutError('Analytical query deadline exceeded') from None
        except psycopg.Error:
            # PostgreSQL error detail can contain query text or private data.
            raise RuntimeError('Analytical query could not be completed') from None
