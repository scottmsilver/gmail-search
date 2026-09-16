"""Derived writers exercised on disposable PostgreSQL with colliding Gmail IDs."""
import os
import uuid
from types import SimpleNamespace

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import pytest

from gmail_search import summarize, propositions


@pytest.fixture(autouse=True)
def _isolated_pg_schema():
    """This module owns its disposable schema; never initialize the dev database."""
    yield None


@pytest.fixture
def derived_db(monkeypatch):
    from gmail_search.auth import write_user
    monkeypatch.setattr(write_user, "_BOOTSTRAP_CACHE", {})
    monkeypatch.setenv("GMS_BOOTSTRAP_EMAIL", "scottmsilver@gmail.com")
    dsn = os.environ.get('GMS_GATEWAY_TEST_DSN')
    if not dsn:
        pytest.skip('GMS_GATEWAY_TEST_DSN must point to disposable PostgreSQL')
    schema = 'owner_derived_' + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        conn.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        conn.execute('''CREATE TABLE users (id text, email text);
            INSERT INTO users VALUES ('bob','scottmsilver@gmail.com');
            CREATE TABLE messages (user_id text, id text, thread_id text DEFAULT 'thread',
            from_addr text DEFAULT '', to_addr text DEFAULT '', subject text, body_text text DEFAULT '',
            body_html text DEFAULT '', history_id bigint DEFAULT 0, raw_json text DEFAULT '{}', labels text DEFAULT '[]', date text DEFAULT '2026-01-01',
            PRIMARY KEY(user_id,id));
            CREATE TABLE attachments (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id text, message_id text, filename text, extracted_text text,
            image_path text, embed_status text, mime_type text DEFAULT 'text/plain', fetch_status text DEFAULT 'ok', size_bytes bigint DEFAULT 1,
            raw_path text, embed_attempts integer DEFAULT 0, embed_error text, embed_last_attempt_at timestamptz);
            CREATE TABLE message_summaries (user_id text, message_id text, summary text, model text,
            created_at timestamptz DEFAULT now(), PRIMARY KEY(user_id,message_id),
            FOREIGN KEY(user_id,message_id) REFERENCES messages(user_id,id));
            CREATE TABLE summary_failures (user_id text, message_id text, model text, error text,
            attempts integer, last_seen timestamptz DEFAULT now(), PRIMARY KEY(user_id,message_id));
            CREATE TABLE propositions (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id text, message_id text, thread_id text, text text, embedding bytea, model text, date text);
            CREATE TABLE prop_processed (user_id text, message_id text, PRIMARY KEY(user_id,message_id));
            CREATE TABLE embeddings (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id text, message_id text, attachment_id bigint, chunk_type text, chunk_text text, embedding bytea, model text)''')
        for owner in ('alice', 'bob'):
            conn.execute("INSERT INTO messages(user_id,id,subject) VALUES (%s,'collision',%s)", (owner, owner))
            conn.execute("INSERT INTO attachments(user_id,message_id,filename,extracted_text) VALUES (%s,'collision','same.txt',%s)", (owner, owner * 50))
        try:
            yield conn
        finally:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


def test_summaries_and_failures_preserve_colliding_owners(derived_db):
    conn = derived_db
    for owner in ('alice', 'bob'):
        summarize._store_summary(conn, 'collision', owner, 'model', user_id=owner)
        summarize._record_summary_failure(conn, 'collision', 'model', owner, user_id=owner)
    summarize._record_summary_failure(conn, 'collision', 'model', 'retry', user_id='alice')
    summarize._clear_summary_failure(conn, 'collision', user_id='alice')
    assert conn.execute('SELECT user_id,attempts FROM summary_failures').fetchall() == [{'user_id': 'bob', 'attempts': 1}]
    assert summarize.get_summary(conn, 'collision', user_id='alice') == 'alice'
    assert summarize.get_summaries_bulk(conn, ['collision'], user_id='bob') == {'collision': 'bob'}
    assert summarize.get_summaries_bulk_meta(conn, ['collision'], model='model', user_id='alice')['collision']['summary'] == 'alice'


def test_summary_queue_and_attachment_context_are_owner_scoped(derived_db):
    conn = derived_db
    conn.execute("INSERT INTO message_summaries(user_id,message_id,summary,model) VALUES ('bob','collision','bob','model')")
    pending = summarize._messages_needing_summary(conn, 'model', None, 'alice')
    assert len(pending) == 1
    assert pending[0]['attachments'] == [{'filename': 'same.txt', 'extracted_text': 'alice' * 50}]


def test_fact_processing_preserves_other_owner(derived_db, monkeypatch):
    monkeypatch.setattr(propositions, 'extract_propositions', lambda *a, **kw: [kw['subject']])
    embedder = SimpleNamespace(model='synthetic', embed_texts_batch=lambda texts: [[1.0, 0.0] for _ in texts])
    for owner in ('bob', 'alice'):
        stats = propositions.propositionize_pending(derived_db, None, None, embedder, user_id=owner, owner=owner)
        assert stats == {'messages': 1, 'facts': 1, 'errors': 0}
    rows = derived_db.execute('SELECT user_id,text FROM propositions ORDER BY user_id').fetchall()
    assert rows == [{'user_id': 'alice', 'text': 'alice'}, {'user_id': 'bob', 'text': 'bob'}]


def test_embedding_pipeline_keeps_attachment_text_with_its_owner(derived_db, monkeypatch):
    from gmail_search.embed import pipeline
    class BorrowedConnection:
        def __getattr__(self, name):
            return getattr(derived_db, name)
        def close(self):
            pass
    monkeypatch.setattr(pipeline, 'get_connection', lambda path: BorrowedConnection())
    monkeypatch.setattr(pipeline, 'check_budget', lambda *a, **kw: (True, 0, 100))
    monkeypatch.setattr(pipeline, 'record_cost', lambda *a, **kw: None)
    embedder = SimpleNamespace(embed_texts_batch=lambda texts: [[1.0, 0.0] for _ in texts])
    config = {'embedding': {'model': 'synthetic'}, 'budget': {'max_usd': 100}}
    for owner in ('alice', 'bob'):
        assert pipeline.run_embedding_pipeline(None, config, embedder, user_id=owner) == 2
    rows = derived_db.execute("SELECT e.user_id,a.user_id AS attachment_owner,e.chunk_text FROM embeddings e JOIN attachments a ON a.id=e.attachment_id ORDER BY e.user_id").fetchall()
    assert len(rows) == 2
    for row in rows:
        assert row['user_id'] == row['attachment_owner']
        assert row['user_id'] * 50 in row['chunk_text']


def test_default_embedding_pipeline_resolves_one_mailbox(derived_db, monkeypatch):
    from gmail_search.embed import pipeline
    class BorrowedConnection:
        def __getattr__(self, name):
            return getattr(derived_db, name)
        def close(self):
            pass
    monkeypatch.setattr(pipeline, 'get_connection', lambda path: BorrowedConnection())
    monkeypatch.setattr(pipeline, 'check_budget', lambda *a, **kw: (True, 0, 100))
    monkeypatch.setattr(pipeline, 'record_cost', lambda *a, **kw: None)
    embedder = SimpleNamespace(embed_texts_batch=lambda texts: [[1.0, 0.0] for _ in texts])
    config = {'embedding': {'model': 'synthetic'}, 'budget': {'max_usd': 100}}
    assert pipeline.run_embedding_pipeline(None, config, embedder) == 2
    assert derived_db.execute('SELECT DISTINCT user_id FROM embeddings').fetchall() == [{'user_id': 'bob'}]
