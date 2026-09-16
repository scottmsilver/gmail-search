"""Real PostgreSQL transcript persistence, isolated from all mail databases."""
import os
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from gmail_search.gateway.registry import AccessDenied


@pytest.fixture
def store():
    from gmail_search.gateway.browser_conversations import BrowserConversations
    dsn = os.environ.get('GMS_GATEWAY_TEST_DSN')
    if not dsn:
        pytest.skip('Synthetic PostgreSQL DSN required')
    name = 'gms_browser_test_'+secrets.token_hex(8)
    with psycopg.connect(dsn,autocommit=True) as db:
        db.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    settings = conninfo_to_dict(dsn)
    settings['dbname'] = name
    target = make_conninfo(**settings)
    with psycopg.connect(target) as db:
        db.execute('CREATE TABLE conversations(id text PRIMARY KEY,user_id text NOT NULL,title text,created_at text DEFAULT now(),updated_at text DEFAULT now())')
        db.execute('CREATE TABLE conversation_messages(id bigserial PRIMARY KEY,conversation_id text REFERENCES conversations(id) ON DELETE CASCADE,seq bigint,role text,parts text,UNIQUE(conversation_id,seq))')
        BrowserConversations.install(db)
    active = {'alice','bob'}
    allowed = {'run'}
    events = [{'seq':1,'event':{'type':'tool_start','name':'search','args':{'query':'receipts'}}},
              {'seq':2,'event':{'type':'text','text':'Private draft'}}]
    def authorize(owner,conversation,run):
        if owner not in active or run not in allowed:
            raise AccessDenied()
        return True
    instance = BrowserConversations(lambda:psycopg.connect(target),is_active=lambda owner:owner in active,
        authorize_persistence=authorize,read_events=lambda owner,conversation,run:events,
        can_edit=lambda owner,conversation: True)
    try:
        yield instance,target,active,allowed
    finally:
        with psycopg.connect(dsn,autocommit=True) as db:
            db.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def test_owner_claim_and_idempotent_rich_answer(store):
    instance,target,_,_ = store
    assert instance.claim('alice','conversation') is True
    with pytest.raises(AccessDenied):
        instance.claim('bob','conversation')
    assert instance.persist('alice','conversation','run','Answer') is True
    assert instance.persist('alice','conversation','run','Answer') is True
    with psycopg.connect(target) as db:
        rows = db.execute('SELECT role,parts::jsonb FROM conversation_messages').fetchall()
        assert len(rows) == 1 and rows[0][0] == 'assistant'
        assert rows[0][1][-1] == {'type':'text','text':'Answer'}
        assert rows[0][1][0]['type'] == 'data-deep-stage'
        assert rows[0][1][0]['data']['payload']['name'] == 'search'
    with pytest.raises(AccessDenied):
        instance.persist('bob','conversation','run','Answer')
    with pytest.raises(AccessDenied):
        instance.persist('alice','conversation','run','Changed answer')


def test_revocation_before_commit_rolls_back_answer(store):
    instance,target,_,allowed = store
    instance.claim('alice','conversation')
    original = instance.authorize_persistence
    calls = 0
    def revoke(owner,conversation,run):
        nonlocal calls
        calls += 1
        if calls == 2:
            allowed.clear()
        return original(owner,conversation,run)
    instance.authorize_persistence = revoke
    with pytest.raises(AccessDenied):
        instance.persist('alice','conversation','run','must not persist')
    with psycopg.connect(target) as db:
        assert db.execute('SELECT count(*) FROM conversation_messages').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM browser_answer_receipts').fetchone()[0] == 0


def test_unknown_or_inactive_conversation_cannot_receive_answer(store):
    instance,target,active,_ = store
    with pytest.raises(AccessDenied):
        instance.persist('alice','missing','run','Answer')
    instance.claim('alice','conversation')
    active.remove('alice')
    with pytest.raises(AccessDenied):
        instance.persist('alice','conversation','run','Answer')
    with psycopg.connect(target) as db:
        assert db.execute('SELECT count(*) FROM conversation_messages').fetchone()[0] == 0


def test_conversation_crud_is_owner_scoped_and_bounded(store):
    instance,target,_,_ = store
    payload = {'title':'Alice chat','messages':[
        {'role':'user','parts':[{'type':'text','text':'hello'}]},
    ]}
    assert instance.save('alice','conversation',payload) is True
    assert instance.list('alice') == [{
        'id':'conversation','title':'Alice chat',
        'created_at':instance.get('alice','conversation')['created_at'],
        'updated_at':instance.get('alice','conversation')['updated_at'],
        'message_count':1,
    }]
    assert instance.get('bob','conversation') is None
    with pytest.raises(AccessDenied):
        instance.save('bob','conversation',payload)
    with pytest.raises(AccessDenied):
        instance.delete('bob','conversation')
    with pytest.raises(ValueError):
        instance.list('alice',101)
    with psycopg.connect(target) as db:
        assert db.execute('SELECT user_id,title FROM conversations').fetchall() == [('alice','Alice chat')]


def test_save_appends_one_user_turn_and_preserves_rich_history(store):
    instance,target,_,_ = store
    first = {'role':'user','parts':[{'type':'text','text':'first question'}]}
    second = {'role':'user','parts':[{'type':'text','text':'second question'}]}
    instance.save('alice','conversation',{'title':'First','messages':[first]})
    instance.persist('alice','conversation','run','Rich answer')
    with psycopg.connect(target) as db:
        rich = db.execute("SELECT parts FROM conversation_messages WHERE role='assistant'").fetchone()[0]

    assert instance.save('alice','conversation',{'title':'Updated','messages':[first,second]}) is True
    conversation = instance.get('alice','conversation')
    assert conversation['title'] == 'Updated'
    assert [row['role'] for row in conversation['messages']] == ['user','assistant','user']
    assert conversation['messages'][1]['parts'][0]['data']['payload']['name'] == 'search'
    with psycopg.connect(target) as db:
        assert db.execute("SELECT parts FROM conversation_messages WHERE role='assistant'").fetchone()[0] == rich

    changed = {'role':'user','parts':[{'type':'text','text':'rewritten question'}]}
    extra = {'role':'user','parts':[{'type':'text','text':'another new question'}]}
    for messages in ([changed,second],[first,second,changed,extra]):
        with pytest.raises(AccessDenied):
            instance.save('alice','conversation',{'title':'Poison','messages':messages})
    assert instance.get('alice','conversation')['title'] == 'Updated'


def test_save_and_delete_require_edit_authorization_at_commit(store):
    instance,target,_,_ = store
    instance.save('alice','conversation',{'messages':[]})

    instance.can_edit = None
    with pytest.raises(AccessDenied):
        instance.save('alice','conversation',{'title':'Denied','messages':[]})
    with pytest.raises(AccessDenied):
        instance.delete('alice','conversation')

    checks = 0
    def revoke_before_commit(owner,conversation):
        nonlocal checks
        checks += 1
        return checks == 1
    instance.can_edit = revoke_before_commit
    with pytest.raises(AccessDenied):
        instance.delete('alice','conversation')
    assert instance.get('alice','conversation') is not None

    checks = 0
    with pytest.raises(AccessDenied):
        instance.save('alice','conversation',{'title':'Denied','messages':[
            {'role':'user','parts':[{'type':'text','text':'must roll back'}]},
        ]})
    assert instance.get('alice','conversation')['messages'] == []

    instance.can_edit = lambda owner,conversation: False
    with pytest.raises(AccessDenied):
        instance.save('alice','conversation',{'title':'Denied','messages':[]})
    with psycopg.connect(target) as db:
        assert db.execute('SELECT title FROM conversations WHERE id=%s',('conversation',)).fetchone()[0] is None


def test_get_rejects_history_beyond_read_byte_bound(store,monkeypatch):
    from gmail_search.gateway import browser_conversations
    instance,_,_,_ = store
    instance.save('alice','conversation',{'messages':[
        {'role':'user','parts':[{'type':'text','text':'bounded'}]},
    ]})
    monkeypatch.setattr(browser_conversations,'_MAX_READ_PARTS_BYTES',8)
    with pytest.raises(AccessDenied):
        instance.get('alice','conversation')


def test_save_rejects_non_utf8_title_as_safe_input_error(store):
    instance,target,_,_ = store
    with pytest.raises(AccessDenied):
        instance.save('alice','conversation',{'title':'bad\ud800','messages':[]})
    with psycopg.connect(target) as db:
        assert db.execute('SELECT count(*) FROM conversations').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_full_controller_persists_real_postgres_transcript(store,tmp_path):
    import asyncio
    from gmail_search.gateway.browser_conversations import compose_browser_runs
    from gmail_search.gateway.capabilities import Capabilities
    from gmail_search.gateway.events import Events
    from gmail_search.gateway.registry import Registry
    from gmail_search.gateway.worker import WorkerController
    from test_browser_runs import Backend
    _,target,active,_ = store
    registry = Registry(tmp_path/'registry',is_active=lambda owner:owner in active)
    capabilities = Capabilities(registry)
    events = Events(capabilities)
    backend = Backend(events,capabilities)
    workers = WorkerController(registry,backend)
    budget = registry.create_budget('alice',1000)
    runs,conversations = compose_browser_runs(workers,events,
        connect=lambda:psycopg.connect(target),is_active=lambda owner:owner in active,
        prepare_input=lambda lease,prompt:backend.prompts.__setitem__(lease.run_id,prompt),
        budget_for=lambda owner:budget,poll_seconds=.01)
    conversations.claim('alice','conversation')
    try:
        run = await runs.start('alice','conversation','find my receipts')
        with pytest.raises(AccessDenied):
            conversations.delete('alice','conversation')
        async with asyncio.timeout(3):
            while runs.snapshot('alice','conversation',run)['state'] not in ('completed','failed'):
                await asyncio.sleep(.01)
        assert runs.snapshot('alice','conversation',run)['state'] == 'completed'
        assert not backend.handles
        with psycopg.connect(target) as db:
            row = db.execute('SELECT parts::jsonb FROM conversation_messages').fetchone()[0]
            assert row[-1]['text'] == 'Answer for find my receipts'
            assert row[0]['data']['payload']['name'] == 'query_emails_batch'
        conversations.save('alice','conversation',{'messages':[
            {'role':'user','parts':[{'type':'text','text':'total those receipts'}]}]})
        followup = await runs.start('alice','conversation','total those receipts')
        async with asyncio.timeout(3):
            while runs.snapshot('alice','conversation',followup)['state'] not in ('completed','failed'):
                await asyncio.sleep(.01)
        assert runs.snapshot('alice','conversation',followup)['state'] == 'completed'
        assert 'Answer for find my receipts' in backend.prompts[followup]
        assert backend.prompts[followup].count('total those receipts') == 1
        assert conversations.delete('alice','conversation') is True
        assert conversations.get('alice','conversation') is None
    finally:
        await runs.close()


def test_receipt_install_upgrades_existing_table_and_is_repeatable(store):
    from gmail_search.gateway.browser_conversations import BrowserConversations
    _,target,_,_=store
    with psycopg.connect(target) as db:
        db.execute('ALTER TABLE public.browser_answer_receipts DISABLE ROW LEVEL SECURITY')
        BrowserConversations.install(db)
        BrowserConversations.install(db)
        assert db.execute("SELECT relrowsecurity FROM pg_class WHERE oid='public.browser_answer_receipts'::regclass").fetchone()==(True,)


def test_receipt_install_rejects_missing_cascade_constraint(store):
    from gmail_search.gateway.browser_conversations import BrowserConversations
    _,target,_,_=store
    with psycopg.connect(target) as db:
        db.execute('ALTER TABLE public.browser_answer_receipts DROP CONSTRAINT browser_answer_receipts_conversation_id_fkey')
        with pytest.raises(ValueError,match='receipt schema'):
            BrowserConversations.install(db)
