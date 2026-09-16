"""Closed typed search inputs and fixed owner-qualified query behavior."""
import pytest
from gmail_search.gateway.search_queries import StructuredFilters, Selection, lexical_query


def test_structured_filters_validate_before_database_work():
    assert StructuredFilters(from_filter='sender').from_filter=='sender'
    for value in ('2026-02-30','not a date','2026-09-15 OR true'):
        with pytest.raises(ValueError):
            StructuredFilters(date_from=value)
    with pytest.raises(ValueError):
        StructuredFilters(from_filter='x'*1001)


def test_lexical_builder_has_fixed_fields_and_rejects_injected_syntax():
    assert lexical_query('facts',('needle',),phrase=False)=='text:needle'
    for tokens in (('subject:foreign',),('needle OR *',),('a"b',),('x'*129,)):
        with pytest.raises(ValueError):
            lexical_query('messages',tokens,phrase=False)
    with pytest.raises(ValueError):
        lexical_query('users',('needle',),phrase=False)
    assert Selection((),True).reason is None


import asyncio
import struct
from test_gateway_search_reader import search_database as search_database, provisioned
from gmail_search.gateway.search_reader import SearchReader, SearchRegistry, SearchProfile, SearchLimits
from gmail_search.gateway.data_admission import DataAdmission


def reader_for(db,*,limits=None):
    owner=db.owners[0];credential=provisioned(db,owner)
    return SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2),admission=DataAdmission(),limits=limits)


@pytest.mark.asyncio
async def test_all_lexical_branches_and_structured_filters_stay_owner_bound(search_database):
    db=search_database;owner=db.owners[0];reader=reader_for(db)
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        await q._session.conn.execute("SET LOCAL plan_cache_mode=force_generic_plan")
        for method in (q.lexical_messages,q.lexical_attachments,q.lexical_facts):
            assert [row.message_id for row in (await method(('needle',))).rows]==['same']
            assert (await method(('foreignonly',))).rows==()
        assert (await q.lexical_messages(('needle',),candidate_ids=())).rows==()
        assert (await q.lexical_messages(('needle',),candidate_ids=('missing',))).rows==()
        assert [row.message_id for row in (await q.structured_candidates(StructuredFilters(from_filter='send',has_attachment=True))).rows]==['same']
        assert (await q.structured_candidates(StructuredFilters(from_filter='foreign'))).rows==()
        assert (await q.structured_candidates(StructuredFilters(date_from='2026-09-16'))).rows==()
        assert (await q.structured_candidates(StructuredFilters(has_attachment=False))).rows==()


@pytest.mark.asyncio
async def test_owner_qualified_hydration_topics_context_and_vector_paging(search_database):
    db=search_database;owner,foreign=db.owners;reader=reader_for(db)
    for who,label in ((owner,'own label'),(foreign,'foreign label')):
        db.conn.execute("INSERT INTO public.topics(user_id,topic_id,label) VALUES(%s,'same-topic',%s)",(who,label))
        db.conn.execute("INSERT INTO public.message_topics(user_id,message_id,topic_id) VALUES(%s,'same','same-topic')",(who,))
        db.conn.execute("INSERT INTO public.thread_summary(user_id,thread_id,subject) VALUES(%s,'thread',%s)",(who,label))
        db.conn.execute("INSERT INTO public.message_summaries(user_id,message_id,summary,model) VALUES(%s,'same',%s,'synthetic')",(who,label))
    db.conn.execute("INSERT INTO public.propositions(user_id,id,message_id,thread_id,text,embedding,model) VALUES(%s,2,'same','thread','second',%s,'fixture+v1'),(%s,3,'same','thread','wrong model',%s,'wrong')",(owner,struct.pack('<ff',0,1),owner,b'x'*40000))
    ids=db.conn.execute('SELECT id FROM public.embeddings ORDER BY id').fetchall()
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        matches=await q.hydrate_embeddings(tuple(row[0] for row in ids))
        assert len(matches.rows)==1 and matches.rows[0].chunk_text=='needle'
        assert (await q.hydrate_messages(('same',))).rows[0].summary=='own label'
        assert (await q.hydrate_threads(('thread',))).rows[0].subject=='own label'
        assert (await q.message_topics(('same',))).rows[0].label=='own label'
        assert (await q.topic_facets(('same',))).rows[0].count==1
        assert (await q.owner_aliases()).rows[0].term=='needle'
        assert (await q.owner_contacts()).rows[0].email=='needle@example.test'
        assert await q.fact_count()==3
        page=await q.fact_vectors_page(limit=1)
        assert page.complete is False and page.next_cursor==1
        assert page.rows[0].embedding==struct.pack('<ff',1,0)
        page=await q.fact_vectors_page(after=page.next_cursor,limit=2)
        assert page.complete and [r.id for r in page.rows]==[2,3]
        assert page.rows[1].embedding is None and page.rows[1].vector_status=='model_mismatch'
        page=await q.restricted_vectors_page(('same',))
        assert len(page.rows)==1 and page.rows[0].vector_status=='ok'
        assert (await q.restricted_vectors_page(('missing',))).rows==()


@pytest.mark.asyncio
async def test_byte_budget_is_explicit_and_never_returns_oversized_row(search_database):
    db=search_database;owner=db.owners[0]
    reader=reader_for(db,limits=SearchLimits(max_page_bytes=50,max_session_bytes=100))
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        result=await q.hydrate_messages(('same',))
        assert result.rows==() and result.complete is False and result.reason=='row_bytes'


@pytest.mark.asyncio
async def test_native_custom_and_generic_scores_ignore_foreign_corpus(search_database):
    import psycopg
    from psycopg import sql
    db=search_database;owner,foreign=db.owners;credential=provisioned(db,owner)
    with psycopg.connect(credential.dsn,autocommit=True) as conn:
        for mode in ('force_custom_plan','force_generic_plan'):
            conn.execute('SELECT set_config(\'plan_cache_mode\',%s,false)',(mode,))
            for table,key,field in (('messages','search_id','subject'),('attachments','id','extracted_text'),('propositions','id','text')):
                literal=sql.Literal(owner).as_string(conn)
                statement=f'SELECT {key},paradedb.score({key}) FROM public.{table} WHERE user_id={literal} AND {key} OPERATOR(pg_catalog.@@@) %s ORDER BY paradedb.score({key}) DESC LIMIT %s'
                before=conn.execute(statement,(field+':needle',10),prepare=True).fetchall()
                assert len(before)==1
                db.conn.execute("UPDATE public.messages SET subject=repeat('needle ',100) WHERE user_id=%s",(foreign,))
                db.conn.execute("UPDATE public.attachments SET extracted_text=repeat('needle ',100) WHERE user_id=%s",(foreign,))
                db.conn.execute("UPDATE public.propositions SET text=repeat('needle ',100) WHERE user_id=%s",(foreign,))
                assert conn.execute(statement,(field+':needle',10),prepare=True).fetchall()==before


@pytest.mark.asyncio
async def test_summary_variant_selects_latest_once_per_message(search_database):
    db=search_database;owner=db.owners[0]
    db.conn.execute('ALTER TABLE public.message_summaries DROP CONSTRAINT message_summaries_pkey')
    db.conn.execute('ALTER TABLE public.message_summaries ADD PRIMARY KEY(user_id,message_id,model)')
    db.conn.execute("INSERT INTO public.message_summaries(user_id,message_id,summary,model,created_at) VALUES(%s,'same','old','a','2025-01-01'),(%s,'same','new','b','2026-01-01')",(owner,owner))
    reader=reader_for(db)
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        rows=await q.hydrate_messages(('same',))
        assert rows.complete and len(rows.rows)==1
        assert rows.rows[0].summary=='new'
        assert rows.rows[0].summary_model=='b'
        assert rows.rows[0].summary_created_at=='2026-01-01'


@pytest.mark.asyncio
async def test_aggregate_transfer_budget_cannot_be_reset_between_operations(search_database):
    db=search_database;owner=db.owners[0]
    reader=reader_for(db,limits=SearchLimits(max_page_bytes=500,max_session_bytes=500))
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        outcomes=[await q.lexical_messages(('needle',)) for _ in range(8)]
        assert outcomes[0].complete and outcomes[0].rows
        assert not outcomes[-1].complete
        assert q._session.bytes<=500


@pytest.mark.asyncio
async def test_same_snapshot_rejects_concurrent_operations(search_database,monkeypatch):
    from gmail_search.gateway.search_reader import _Session
    db=search_database;owner=db.owners[0];reader=reader_for(db)
    started=asyncio.Event();release=asyncio.Event();original=_Session._timeout
    async def gated(self):
        started.set();await release.wait();await original(self)
    monkeypatch.setattr(_Session,'_timeout',gated)
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        first=asyncio.create_task(q.fact_count());await started.wait()
        with pytest.raises(RuntimeError,match='Concurrent'):
            await q.fact_count()
        release.set();assert await first==1


@pytest.mark.asyncio
async def test_nonfinite_database_score_is_not_returned_as_typed_numeric(search_database):
    db=search_database;owner=db.owners[0];reader=reader_for(db)
    db.conn.execute("UPDATE public.contact_frequency SET score='NaN'::float8 WHERE user_id=%s",(owner,))
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        with pytest.raises(RuntimeError,match='numeric'):
            await q.owner_contacts()
