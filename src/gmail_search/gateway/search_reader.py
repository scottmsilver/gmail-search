"""Immutable internal search sessions with owned cancellation and shared admission."""
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import asyncio
import hashlib
import inspect
import json
import math
from types import MappingProxyType

import numpy as np
import psycopg
from psycopg.conninfo import conninfo_to_dict

from .bounded_fetch import bounded_rows
from .search_queries import BoundSearchQueries, MessageCandidate, Selection
from .partition_profiles import NUMERIC_OWNER_PARTITIONS_V1 as NUMERIC, TEXT_OWNER_PARTITIONS_V1 as TEXT, PartitionSchemaProfile, require_profile


def _columns(text='',int8='',float8='',bytea=''):
    return MappingProxyType({**{key:'text' for key in ('user_id '+text).split()},
        **{key:'int8' for key in int8.split()},**{key:'float8' for key in float8.split()},**{key:'bytea' for key in bytea.split()}})


# Independently closed; future analytical schema additions never widen this role.
SEARCH_COLUMNS=MappingProxyType({
    'messages':_columns('id thread_id from_addr to_addr subject body_text date labels',int8='search_id'),
    'attachments':_columns('message_id filename mime_type extracted_text fetch_status',int8='id size_bytes'),
    'embeddings':_columns('message_id chunk_type chunk_text model',int8='id attachment_id',bytea='embedding'),
    'thread_summary':_columns('thread_id subject participants all_from_addrs all_labels date_first date_last',int8='message_count'),
    'message_summaries':_columns('message_id summary model created_at'),
    'topics':_columns('topic_id parent_id label'),
    'message_topics':_columns('message_id topic_id'),
    'propositions':_columns('message_id thread_id text model',int8='id',bytea='embedding'),
    'term_aliases':_columns('term expansions'),
    'contact_frequency':_columns('email',float8='score'),
})
_TEXT_SEARCH_COLUMNS = MappingProxyType({**SEARCH_COLUMNS,
    'messages': MappingProxyType({name:kind for name,kind in SEARCH_COLUMNS['messages'].items() if name != 'search_id'})})


def search_columns(profile=NUMERIC):
    require_profile(profile)
    return SEARCH_COLUMNS if profile is NUMERIC else _TEXT_SEARCH_COLUMNS


SEARCH_FUNCTIONS=MappingProxyType({
    'paradedb.search_with_parse(anyelement,text)':('boolean','c','search_with_parse_wrapper','$libdir/pg_search','i',False,False,'s','f'),
    'paradedb.score(anyelement)':('real','c','paradedb_score_from_relation_wrapper','$libdir/pg_search','s',False,False,'s','f'),
    'paradedb.with_index(regclass,paradedb.searchqueryinput)':('paradedb.searchqueryinput','c','with_index_wrapper','$libdir/pg_search','i',False,False,'s','f'),
    'paradedb.parse_with_field(paradedb.fieldname,text,boolean,boolean)':('paradedb.searchqueryinput','c','parse_with_field_bfn_wrapper','$libdir/pg_search','i',False,False,'s','f'),
})

_TEXT_SEARCH_FUNCTIONS = MappingProxyType({**SEARCH_FUNCTIONS,
    'paradedb.terms_with_operator(paradedb.fieldname,text,anyelement,boolean)':
        ('paradedb.searchqueryinput','c','terms_with_operator_wrapper','$libdir/pg_search','i',False,False,'s','f'),
})


def search_functions(profile=NUMERIC):
    require_profile(profile)
    return SEARCH_FUNCTIONS if profile is NUMERIC else _TEXT_SEARCH_FUNCTIONS


ROLE_SETTINGS=(('search_path','pg_catalog'),('default_transaction_read_only','on'),('statement_timeout','10s'),
    ('lock_timeout','1s'),('idle_in_transaction_session_timeout','10s'),('work_mem','4MB'),('temp_file_limit','64MB'),('max_parallel_workers_per_gather','0'),('plan_cache_mode','force_custom_plan'))


def search_role(owner_id, *, profile=NUMERIC):
    require_profile(profile)
    if not isinstance(owner_id,str) or not owner_id or len(owner_id)>2048 or '\x00' in owner_id:
        raise ValueError('Invalid search owner')
    try:encoded=owner_id.encode('utf-8')
    except UnicodeError:raise ValueError('Invalid search owner') from None
    return ('gms_search_' if profile is NUMERIC else 'gms_search_text_')+hashlib.sha256(encoded).hexdigest()[:40]


def reader_binding(owner_id, *, profile=NUMERIC):
    search_role(owner_id, profile=profile)
    if profile is TEXT:
        return 'gmail-search internal search reader text-owner-partitions-v1 owner='+owner_id
    return 'gmail-search internal search reader v1 owner='+owner_id


@dataclass(frozen=True)
class SearchCredential:
    owner_id: str
    dsn: str=field(repr=False)
    schema_profile: PartitionSchemaProfile=NUMERIC

    def __post_init__(self):
        try:parsed=conninfo_to_dict(self.dsn)
        except Exception:raise ValueError('Invalid search connection configuration') from None
        if parsed.get('user')!=search_role(self.owner_id, profile=self.schema_profile) or parsed.get('service') or parsed.get('options'):
            raise ValueError('Search login does not match its fixed owner')


@dataclass(frozen=True)
class SearchProfile:
    embedding_model: str
    fact_model_tag: str
    dimensions: int
    schema_profile: PartitionSchemaProfile=NUMERIC
    # The `embeddings.model` tag the corpus was written under, when it differs
    # from the API model that embeds queries (same vectors, different name).
    stored_embedding_tag: str|None=None

    @property
    def embedding_tag(self):
        return self.stored_embedding_tag or self.embedding_model

    def __post_init__(self):
        require_profile(self.schema_profile)
        if any(not isinstance(value,str) or not value or len(value)>200 or '\x00' in value for value in (self.embedding_model,self.fact_model_tag,self.embedding_tag)):
            raise ValueError('Invalid trusted search model profile')
        if type(self.dimensions) is not int or not 1<=self.dimensions<=4096:
            raise ValueError('Invalid trusted vector dimensions')


@dataclass(frozen=True)
class SearchLimits:
    deadline_seconds: float=30
    statement_seconds: float=10
    max_page_bytes: int=4*1024*1024
    max_session_bytes: int=128*1024*1024
    max_session_rows: int=200_000

    def __post_init__(self):
        for name,maximum in (('deadline_seconds',30),('statement_seconds',10)):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=maximum:
                raise ValueError('Invalid search deadline')
        for name,maximum in (('max_page_bytes',4*1024*1024),('max_session_bytes',128*1024*1024),('max_session_rows',200_000)):
            value=getattr(self,name)
            if type(value) is not int or not 0<value<=maximum:raise ValueError('Invalid search resource budget')


class SearchRegistry:
    def __init__(self,credentials,*,is_active):
        if any(not isinstance(value,SearchCredential) or key!=value.owner_id for key,value in credentials.items()):
            raise ValueError('Search registry owner mismatch')
        self._credentials=MappingProxyType(dict(credentials));self._is_active=is_active

    def credential(self,owner_id):
        value=self._credentials.get(owner_id)
        if value is None or not self._is_active(owner_id):raise PermissionError('Mailbox search is unavailable')
        return value


async def _drain(task):
    """Wait for acknowledgement despite repeated cancellation of our caller."""
    waiter=asyncio.ensure_future(asyncio.gather(task,return_exceptions=True))
    interrupted=False
    while not waiter.done():
        try:await asyncio.shield(waiter)
        except asyncio.CancelledError:interrupted=True
    return interrupted


# Concurrent search sessions. Each may buffer up to SearchLimits.max_session_bytes
# (128 MB), so 16 bounds worst-case memory near 2 GB; raised from 8/2 so a run's
# parallel subagents are not refused (2026-09-24).
MAX_SEARCH_GLOBAL=16
MAX_SEARCH_PER_OWNER=8


class SearchReader:
    def __init__(self,registry,*,profile,admission,limits=None):
        self.registry=registry;self.profile=profile;self.admission=admission;self.limits=limits or SearchLimits()
        if admission.global_concurrency>MAX_SEARCH_GLOBAL or admission.owner_concurrency>MAX_SEARCH_PER_OWNER:
            raise ValueError('Search admission exceeds qualified profile')

    @asynccontextmanager
    async def session(self,owner_id,*,deadline,check_active):
        if type(deadline) not in (int,float) or not math.isfinite(deadline):raise ValueError('Invalid absolute search deadline')
        credential=self.registry.credential(owner_id)
        if credential.schema_profile is not self.profile.schema_profile:
            raise PermissionError('Search credential schema profile mismatch')
        effective=min(deadline,asyncio.get_running_loop().time()+self.limits.deadline_seconds)
        if effective<=asyncio.get_running_loop().time():raise TimeoutError('Search deadline exceeded')
        lease=self.admission.acquire(owner_id)
        session=_Session(self,credential,effective,check_active)
        parent=asyncio.current_task()
        watch=asyncio.create_task(session.watch(parent))
        try:
            async with asyncio.timeout_at(effective):
                await session.owned(session.open())
                yield BoundSearchQueries(session)
                await session.check()
        except psycopg.errors.QueryCanceled:
            raise TimeoutError('Search statement deadline exceeded') from None
        except psycopg.Error:
            raise RuntimeError('Search database operation failed') from None
        except asyncio.CancelledError:
            if session.expired:raise TimeoutError('Search deadline exceeded') from None
            if session.revoked:raise PermissionError('Mailbox search was revoked') from None
            raise
        finally:
            session.finished=True
            watch.cancel()
            interrupted=await _drain(watch)
            close=asyncio.create_task(session.close())
            interrupted=(await _drain(close)) or interrupted
            # A failed close is quarantined: do not release capacity while its
            # database resource might still be live.
            close.result()
            lease.release()
            if interrupted:raise asyncio.CancelledError


class _Session:
    def __init__(self,reader,credential,deadline,check_active):
        self.reader=reader;self.credential=credential;self.owner_id=credential.owner_id
        self.profile=reader.profile;self.deadline=deadline;self.check_active=check_active
        self.conn=None;self.jobs=set();self.finished=False;self.revoked=False;self.expired=False
        self.bytes=0;self.rows=0;self.serial=0

    async def check(self):
        if self.finished:raise RuntimeError('Search snapshot is closed')
        self.reader.registry.credential(self.owner_id)
        result=self.check_active()
        if inspect.isawaitable(result):result=await result
        if result is False:raise PermissionError('Mailbox search was revoked')
        if asyncio.get_running_loop().time()>=self.deadline:raise TimeoutError('Search deadline exceeded')

    async def watch(self,parent):
        try:
            while True:
                await self.check()
                await asyncio.sleep(.05)
        except asyncio.CancelledError:raise
        except Exception as exc:
            self.expired=isinstance(exc,TimeoutError)
            self.revoked=not self.expired
            parent.cancel()

    async def owned(self,coroutine):
        if self.jobs:
            coroutine.close()
            raise RuntimeError('Concurrent operations on one search snapshot are unsupported')
        task=asyncio.create_task(coroutine);self.jobs.add(task)
        try:return await asyncio.shield(task)
        except BaseException:
            task.cancel()
            await _drain(task)
            raise
        finally:self.jobs.discard(task)

    async def open(self):
        await self.check()
        # Qualified pg_search maintenance policy: never promote recurring
        # statements to generic prepared plans; keep role and session custom.
        self.conn=await psycopg.AsyncConnection.connect(self.credential.dsn,autocommit=True,connect_timeout=3,application_name='gms-internal-search',prepare_threshold=None)
        if self.conn.info.server_version//10000!=16:raise PermissionError('Search database version is unavailable')
        cursor=await self.conn.execute('SELECT session_user,current_user,current_setting(\'temp_file_limit\'),current_setting(\'standard_conforming_strings\'),current_setting(\'plan_cache_mode\')')
        if await cursor.fetchone()!=(search_role(self.owner_id, profile=self.profile.schema_profile),search_role(self.owner_id, profile=self.profile.schema_profile),'64MB','on','force_custom_plan'):
            raise PermissionError('Search database identity is unavailable')
        await self.conn.execute('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for setting,value in (('search_path','pg_catalog'),('row_security','on'),('statement_timeout','10000'),('lock_timeout','1000'),('work_mem','4MB'),('max_parallel_workers_per_gather','0'),('cursor_tuple_fraction','1'),('plan_cache_mode','force_custom_plan')):
            await self.conn.execute('SELECT set_config(%s,%s,true)',(setting,value))
        from .provision_search_reader import _checks
        checks=_checks(self.owner_id, profile=self.profile.schema_profile)
        try:
            statement,params=next(checks)
            while True:
                cursor=await self.conn.execute(statement,params)
                statement,params=checks.send(await cursor.fetchall())
        except StopIteration:pass
        except ValueError:raise PermissionError('Search database profile is unavailable') from None
        await self.check()

    async def close(self):
        for job in tuple(self.jobs):
            job.cancel()
            await _drain(job)
        if self.conn is not None:
            try:
                await self.conn.rollback()
            except psycopg.Error:pass
            finally:await self.conn.close()
            if not self.conn.closed:raise RuntimeError('Search database cleanup was not acknowledged')

    async def count(self,statement,params):
        async def operation():
            await self.check();await self._timeout()
            cursor=await self.conn.execute(statement,params)
            result=await cursor.fetchone()
            await self.check()
            if not result or type(result[0]) is not int or result[0]<0:raise RuntimeError('Invalid search count')
            return result[0]
        return await self.owned(operation())

    async def _timeout(self):
        remaining=self.deadline-asyncio.get_running_loop().time()
        if remaining<=0:raise TimeoutError('Search deadline exceeded')
        milliseconds=max(1,int(min(remaining,self.reader.limits.statement_seconds)*1000))
        await self.conn.execute('SELECT set_config(\'statement_timeout\',%s,true)',(str(milliseconds),))

    def _decode_row(self,payload):
        values=json.loads(payload)
        if values.pop('owner_id',None)!=self.owner_id:raise PermissionError('Search row owner mismatch')
        if ('score' in values and (type(values['score']) not in (int,float) or not math.isfinite(values['score']))) or any(isinstance(value,float) and not math.isfinite(value) for value in values.values()):
            raise RuntimeError('Invalid search numeric result')
        if 'embedding' in values and values['embedding'] is not None:
            vector=bytes.fromhex(values['embedding'])
            if len(vector)!=self.profile.dimensions*4:raise RuntimeError('Invalid search vector size')
            if not np.isfinite(np.frombuffer(vector,dtype='<f4')).all():
                values['embedding']=None;values['vector_status']='invalid_vector'
            else:values['embedding']=vector
        return values

    def _stop_reason(self,payload,accepted,limit,page_bytes,budget):
        """Why this row ends the page, or None to accept it."""
        if accepted>=limit:return 'row_limit'
        if payload is None:return 'row_bytes'
        if self.rows>=self.reader.limits.max_session_rows:return 'session_budget'
        if len(payload.encode('utf-8'))+page_bytes>budget:return 'page_bytes'
        return None

    async def read_message_ids(self,statement,params,limit):
        """IDs from `statement` (columns owner_id, message_id) as one aggregated
        row. A per-row JSON document per ID made 21k candidates cost ~1 s."""
        async def operation():
            await self.check();await self._timeout()
            budget=min(self.reader.limits.max_page_bytes,self.reader.limits.max_session_bytes-self.bytes)
            if budget<=0 or self.rows>=self.reader.limits.max_session_rows:
                return Selection((),False,'session_budget')
            bounded=('SELECT CASE WHEN octet_length(ids::text)<=%s THEN ids::text ELSE NULL END,owners_ok FROM '
                '(SELECT json_agg(gms_row.message_id ORDER BY gms_row.message_id) AS ids,'
                'bool_and(gms_row.owner_id=%s) AS owners_ok FROM ('+statement+' LIMIT %s) gms_row) gms_agg')
            cursor=await self.conn.execute(bounded,(budget,self.owner_id,*params,limit+1))
            payload,owners_ok=await cursor.fetchone()
            await self.check()
            if owners_ok is False:raise PermissionError('Search row owner mismatch')
            if payload is None and owners_ok is not None:return Selection((),False,'page_bytes')
            ids=json.loads(payload) if payload is not None else []
            if not isinstance(ids,list) or any(not isinstance(value,str) for value in ids):
                raise RuntimeError('Invalid search candidate list')
            reason='row_limit' if len(ids)>limit else None
            ids=ids[:limit]
            self.bytes+=len(payload or '');self.rows+=len(ids)
            return Selection(tuple(MessageCandidate(value) for value in ids),reason is None,reason)
        return await self.owned(operation())

    async def read(self,statement,params,row_type,limit,*,cursor=None):
        async def operation():
            await self.check();await self._timeout()
            budget=min(self.reader.limits.max_page_bytes,self.reader.limits.max_session_bytes-self.bytes)
            if budget<=0 or self.rows>=self.reader.limits.max_session_rows:
                return Selection((),False,'session_budget')
            bounded=('SELECT CASE WHEN octet_length(row_to_json(gms_row)::text)<=%s '
                'THEN row_to_json(gms_row)::text ELSE NULL END FROM ('+statement+' LIMIT %s) gms_row')
            output=[];page_bytes=0;reason=None
            self.serial+=1
            async with self.conn.cursor(name='search_'+str(self.serial)) as db_cursor:
                await db_cursor.execute(bounded,(budget,*params,limit+1))
                # Liveness is checked per batch (and below), never per row.
                async with bounded_rows(db_cursor,row_byte_cap=budget,between_batches=self.check) as rows:
                    async for (payload,) in rows:
                        reason=self._stop_reason(payload,len(output),limit,page_bytes,budget)
                        if reason:break
                        size=len(payload.encode('utf-8'))
                        output.append(row_type(**self._decode_row(payload)));page_bytes+=size;self.bytes+=size;self.rows+=1
            await self.check()
            complete=reason is None
            next_cursor=getattr(output[-1],cursor) if output and not complete and cursor and reason in ('row_limit','page_bytes') else None
            return Selection(tuple(output),complete,reason,next_cursor)
        return await self.owned(operation())
