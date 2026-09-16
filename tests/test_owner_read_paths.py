"""Colliding Gmail IDs must not join another owner's metadata or content."""
import numpy as np
import pytest

from test_owner_ingestion import ingestion as ingestion_fixture
from test_gateway_database_integration import database as database_fixture

database = database_fixture
ingestion = ingestion_fixture


@pytest.fixture
def colliding(ingestion):
    conn, (alice, bob) = ingestion
    for owner in (alice, bob):
        conn.execute("INSERT INTO messages(id,user_id,thread_id,subject,body_text,from_addr,date,labels) VALUES('same',%s,'thread',%s,%s,%s,'2026-01-01','[]')", (owner, owner, owner, owner))
        conn.execute("INSERT INTO thread_summary(thread_id,user_id,subject,participants,message_count,date_first,date_last) VALUES('thread',%s,%s,'[]',1,'2026-01-01','2026-01-01')", (owner,owner))
        conn.execute("INSERT INTO message_summaries(message_id,user_id,summary,model,created_at) VALUES('same',%s,%s,'model','2026-01-01')", (owner,owner))
    attachment = conn.execute("INSERT INTO attachments(user_id,message_id,filename) VALUES(%s,'same','bob-file') RETURNING id", (bob,)).fetchone()['id']
    eids = []
    for owner in (alice,bob):
        eids.append(conn.execute("INSERT INTO embeddings(user_id,message_id,chunk_type,chunk_text,embedding) VALUES(%s,'same','message',%s,%s) RETURNING id", (owner,owner,np.zeros(3072,dtype=np.float32).tobytes())).fetchone()['id'])
    return conn,alice,bob,attachment,eids


def test_attachment_filters_use_same_owner_parent(colliding):
    from gmail_search.server import _thread_ids_matching_filters
    conn,alice,bob,_,_ = colliding
    assert _thread_ids_matching_filters(conn,['m.user_id=%s'],[alice],True,'date_desc',10) == []
    assert _thread_ids_matching_filters(conn,['m.user_id=%s'],[alice],False,'date_desc',10) == ['thread']
    assert _thread_ids_matching_filters(conn,['m.user_id=%s'],[bob],True,'date_desc',10) == ['thread']


def test_inbox_summary_joins_keep_mailbox_owner(colliding):
    from gmail_search.server import _inbox_rows
    conn,alice,_,_,_ = colliding
    rows = _inbox_rows(conn,'true',(),10,0,user_id=alice)
    assert len(rows) == 1
    assert rows[0]['summary'] == alice


def test_embedding_metadata_rejects_foreign_ids_and_joins(colliding,monkeypatch):
    from gmail_search.search import engine
    conn,alice,_,_,eids = colliding
    class Borrowed:
        def execute(self,*args,**kwargs): return conn.execute(*args,**kwargs)
        def close(self): pass
    monkeypatch.setattr(engine,'get_connection',lambda _:Borrowed())
    search = object.__new__(engine.SearchEngine)
    search.db_path = None
    search.user_id = alice
    rows = search._fetch_embedding_rows(eids)
    assert list(rows) == [eids[0]]
    assert rows[eids[0]]['subject'] == alice


def test_clustering_cannot_use_foreign_subject_for_same_id(colliding):
    from gmail_search.store.db import _load_message_embeddings
    conn,alice,_,_,_ = colliding
    from gmail_search.store.db import _compat_row_factory
    conn.row_factory = _compat_row_factory
    loaded = _load_message_embeddings(conn,user_id=alice)
    assert loaded['subjects'] == [alice]


def test_clustering_omitted_owner_resolves_one_bootstrap_mailbox(colliding, monkeypatch):
    from gmail_search.auth import write_user
    from gmail_search.store.db import _load_message_embeddings, _compat_row_factory
    conn, alice, _, _, _ = colliding
    conn.row_factory = _compat_row_factory
    monkeypatch.setattr(write_user, 'get_bootstrap_user_id', lambda _: alice)
    assert _load_message_embeddings(conn)['subjects'] == [alice]
