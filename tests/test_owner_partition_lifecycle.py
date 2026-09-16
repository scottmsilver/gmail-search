"""Complete canonical schema, attach migration, startup reapply and new-owner CRUD."""
import importlib.util
import os
from pathlib import Path
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from gmail_search.gateway.partitions import provision_owner_partitions, verify_owner_partitions, remove_empty_owner_partitions

ROOT = Path(__file__).parents[1]


@pytest.fixture
def full_schema():
    dsn = os.getenv('GMS_TEST_PG_DSN')
    if not dsn:
        pytest.skip('Explicit synthetic ParadeDB DSN required')
    cfg = conninfo_to_dict(dsn)
    assert (cfg.get('host'),cfg.get('port'),cfg.get('dbname'),cfg.get('user')) == ('127.0.0.1','55440','postgres','postgres')
    assert not set(cfg) & {'hostaddr','service','options'}
    name = 'gms_owner_partitions_test_' + secrets.token_hex(10)
    with psycopg.connect(dsn,autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    try:
        with psycopg.connect(make_conninfo(dsn,dbname=name),autocommit=True) as conn:
            conn.execute((ROOT/'src/gmail_search/store/pg_schema.sql').read_text())
            conn.execute("INSERT INTO users(id,email) VALUES('alice','alice@example.test'),('bob','bob@example.test')")
            conn.execute("INSERT INTO messages(id,user_id,thread_id,from_addr,to_addr,date,subject,body_text) VALUES('same','alice','thread','sender','recipient','2026-01-01','needle','alpha')")
            conn.execute("INSERT INTO attachments(user_id,message_id,filename,mime_type,extracted_text) VALUES('alice','same','file','text/plain','alpha')")
            conn.execute("INSERT INTO propositions(user_id,message_id,text,embedding,model) VALUES('alice','same','alpha',%s,'synthetic')",(b'\x00\x01',))
            conn.execute("INSERT INTO embeddings(user_id,message_id,attachment_id,chunk_type,embedding,model) VALUES('alice','same',1,'attachment_text',%s,'synthetic')",(b'\x01\x02',))
            yield conn
    finally:
        with psycopg.connect(dsn,autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def test_full_schema_attach_reapply_new_owner_and_foreign_keys(full_schema):
    conn = full_schema
    spec = importlib.util.spec_from_file_location('full_owner_partition_migration', ROOT/'deploy/public/migrate_owner_partitions.py')
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.migrate_owner_partitions(conn,owner_id='alice') is True
    verify_owner_partitions(conn,'alice')
    conn.execute((ROOT/'src/gmail_search/store/pg_schema.sql').read_text())
    verify_owner_partitions(conn,'alice')
    provision_owner_partitions(conn,'bob')
    # Same Gmail and numeric identities belong to different users.
    conn.execute("INSERT INTO messages(id,user_id,search_id,thread_id,from_addr,to_addr,date,subject,body_text) VALUES('same','bob',1,'thread','b','b','2026-01-01','foreignonly','beta')")
    conn.execute("INSERT INTO attachments(id,user_id,message_id,filename,mime_type) VALUES(1,'bob','same','file','text/plain')")
    conn.execute("INSERT INTO propositions(id,user_id,message_id,text,model) VALUES(1,'bob','same','beta','synthetic')")
    conn.execute("INSERT INTO embeddings(user_id,message_id,attachment_id,chunk_type,embedding,model) VALUES('bob','same',1,'attachment_text',%s,'synthetic')",(b'\x03',))
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute("INSERT INTO embeddings(user_id,message_id,attachment_id,chunk_type,embedding,model) VALUES('bob','missing',1,'attachment_text',%s,'synthetic')",(b'\x04',))
    assert conn.execute("SELECT id FROM messages WHERE user_id='alice' AND search_id @@@ 'subject:needle'").fetchall() == [('same',)]
    assert conn.execute("SELECT id FROM messages WHERE user_id='alice' AND search_id @@@ 'subject:foreignonly'").fetchall() == []
    for owner in ('alice','bob'):
        assert conn.execute('SELECT count(*) FROM attachments WHERE user_id=%s',(owner,)).fetchone() == (1,)
        assert conn.execute('SELECT count(*) FROM propositions WHERE user_id=%s',(owner,)).fetchone() == (1,)
    # Restricting embeddings remain protected when a mailbox deletion is attempted.
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute("DELETE FROM messages WHERE user_id='bob'")
    conn.execute("DELETE FROM embeddings WHERE user_id='bob'")
    conn.execute("DELETE FROM attachments WHERE user_id='bob'")
    conn.execute("DELETE FROM messages WHERE user_id='bob'")
    remove_empty_owner_partitions(conn,'bob')
    conn.execute("DELETE FROM users WHERE id='bob'")
    verify_owner_partitions(conn,'alice')
    assert conn.execute("SELECT count(*) FROM messages WHERE user_id='bob'").fetchone() == (0,)
    assert conn.execute("SELECT count(*) FROM messages WHERE user_id='alice'").fetchone() == (1,)
