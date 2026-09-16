"""Schema migration qualification on an isolated database; no application callers."""
import importlib.util
import os
from pathlib import Path
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest


@pytest.fixture
def migration():
    path = Path(__file__).parents[1] / 'deploy/public/migrate_owner_keys.py'
    spec = importlib.util.spec_from_file_location('owner_key_schema_migration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def legacy_schema():
    dsn = os.environ.get('GMS_GATEWAY_TEST_DSN')
    if not dsn:
        pytest.skip('Requires disposable PostgreSQL test instance')
    name = 'gms_owner_keys_test_' + secrets.token_hex(8)
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    config = conninfo_to_dict(dsn)
    config['dbname'] = name
    try:
        with psycopg.connect(make_conninfo(**config), autocommit=True) as conn:
            conn.execute('''
                CREATE TABLE users (id text PRIMARY KEY);
                INSERT INTO users VALUES ('alice'), ('bob');
                CREATE TABLE messages (id text PRIMARY KEY, user_id text REFERENCES users(id), body_text text);
                CREATE TABLE topics (user_id text REFERENCES users(id), topic_id text, PRIMARY KEY(user_id,topic_id));
                CREATE TABLE attachments (id bigserial PRIMARY KEY, user_id text REFERENCES users(id),
                    message_id text REFERENCES messages(id), filename text, raw_path text, UNIQUE(message_id,filename));
                CREATE TABLE embeddings (id bigserial PRIMARY KEY, user_id text REFERENCES users(id),
                    message_id text REFERENCES messages(id), attachment_id bigint REFERENCES attachments(id), embedding bytea);
                CREATE TABLE message_summaries (message_id text PRIMARY KEY REFERENCES messages(id),
                    user_id text REFERENCES users(id), summary text);
                CREATE TABLE summary_failures (message_id text PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                    user_id text REFERENCES users(id), error text);
                CREATE TABLE message_topics (message_id text REFERENCES messages(id), topic_id text,
                    user_id text REFERENCES users(id), PRIMARY KEY(message_id,topic_id),
                    FOREIGN KEY(user_id,topic_id) REFERENCES topics(user_id,topic_id));
                INSERT INTO messages VALUES ('same','alice','original body');
                INSERT INTO topics VALUES ('alice','topic'), ('bob','topic');
                INSERT INTO attachments(user_id,message_id,filename,raw_path) VALUES ('alice','same','file','original path');
                INSERT INTO embeddings(user_id,message_id,attachment_id,embedding) VALUES ('alice','same',1,'\\x0102');
                INSERT INTO message_summaries VALUES ('same','alice','original summary');
                INSERT INTO summary_failures VALUES ('same','alice','original failure');
                INSERT INTO message_topics VALUES ('same','topic','alice');
                ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
                CREATE POLICY legacy_reader ON messages USING (true);
            ''')
            yield conn
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def test_migration_preserves_content_and_allows_owner_collisions_idempotently(legacy_schema, migration):
    conn = legacy_schema
    assert migration.migrate_owner_keys(conn) is True
    original = conn.execute('SELECT id,user_id,body_text,search_id FROM messages').fetchone()
    assert original[:3] == ('same','alice','original body')
    assert type(original[3]) is int
    assert conn.execute('SELECT id,raw_path FROM attachments').fetchall() == [(1,'original path')]
    assert conn.execute('SELECT id,embedding FROM embeddings').fetchall() == [(1,b'\x01\x02')]
    assert conn.execute('SELECT message_id,user_id,summary FROM message_summaries').fetchall() == [('same','alice','original summary')]
    assert conn.execute('SELECT relrowsecurity FROM pg_class WHERE oid=\'public.messages\'::regclass').fetchone() == (True,)
    assert conn.execute('SELECT count(*) FROM pg_policy WHERE polrelid=\'public.messages\'::regclass').fetchone() == (1,)
    assert migration.migrate_owner_keys(conn) is False
    assert conn.execute('SELECT search_id FROM messages').fetchone() == (original[3],)
    conn.execute("INSERT INTO messages(id,user_id,body_text) VALUES ('same','bob','Bob body')")
    conn.execute("INSERT INTO attachments(user_id,message_id,filename) VALUES ('bob','same','file')")
    conn.execute("INSERT INTO message_summaries VALUES ('same','bob','Bob summary')")
    conn.execute("INSERT INTO summary_failures VALUES ('same','bob','Bob failure')")
    conn.execute("INSERT INTO message_topics VALUES ('same','topic','bob')")
    assert conn.execute('SELECT count(*),count(DISTINCT search_id) FROM messages').fetchone() == (2,2)
    assert conn.execute('SELECT count(*) FROM attachments').fetchone() == (2,)
    assert conn.execute('SELECT count(*) FROM pg_constraint WHERE NOT convalidated').fetchone() == (0,)


def test_target_fks_reject_cross_owner_and_cross_message_attachments(legacy_schema, migration):
    conn = legacy_schema
    migration.migrate_owner_keys(conn)
    for statement in [
        "INSERT INTO attachments(user_id,message_id,filename) VALUES ('bob','same','foreign')",
        "INSERT INTO embeddings(user_id,message_id,attachment_id) VALUES ('bob','same',1)",
        "INSERT INTO message_summaries VALUES ('same','bob','foreign')",
        "INSERT INTO summary_failures VALUES ('same','bob','foreign')",
        "INSERT INTO message_topics VALUES ('same','topic','bob')",
    ]:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            conn.execute(statement)
    conn.execute("INSERT INTO messages(id,user_id) VALUES ('other','alice')")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute("INSERT INTO embeddings(user_id,message_id,attachment_id) VALUES ('alice','other',1)")


@pytest.mark.parametrize('table', ['messages','attachments','embeddings','message_summaries','summary_failures','message_topics'])
def test_null_owner_preflight_refuses_without_schema_change(legacy_schema, migration, table):
    conn = legacy_schema
    conn.execute(sql.SQL('UPDATE {} SET user_id=NULL').format(sql.Identifier(table)))
    with pytest.raises(ValueError, match='null owner'):
        migration.migrate_owner_keys(conn)
    assert conn.execute("SELECT 1 FROM pg_attribute WHERE attrelid='messages'::regclass AND attname='search_id'").fetchone() is None


@pytest.mark.parametrize('table', ['attachments','embeddings','message_summaries','summary_failures','message_topics'])
def test_cross_owner_existing_child_refuses(legacy_schema, migration, table):
    conn = legacy_schema
    conn.execute(sql.SQL("UPDATE {} SET user_id='bob'").format(sql.Identifier(table)))
    with pytest.raises(ValueError, match='owner|orphan'):
        migration.migrate_owner_keys(conn)


@pytest.mark.parametrize('extra', [
    'CREATE INDEX messages_bm25_idx ON messages(id)',
    'CREATE TABLE unknown_child (message_id text REFERENCES messages(id))',
    'CREATE UNIQUE INDEX unexpected_global_key ON messages(id)',
])
def test_unqualified_dependencies_refuse(legacy_schema, migration, extra):
    legacy_schema.execute(extra)
    with pytest.raises(ValueError, match='BM25|dependency|index'):
        migration.migrate_owner_keys(legacy_schema)


def test_renamed_bm25_access_method_index_is_refused(legacy_schema, migration):
    if not legacy_schema.execute("SELECT 1 FROM pg_available_extensions WHERE name='pg_search'").fetchone():
        pytest.skip('Real renamed BM25 qualification requires the ParadeDB extension')
    legacy_schema.execute('CREATE EXTENSION IF NOT EXISTS pg_search')
    legacy_schema.execute("CREATE INDEX custom_search_index ON messages USING bm25(id,body_text) WITH (key_field='id')")
    with pytest.raises(ValueError, match='BM25'):
        migration.migrate_owner_keys(legacy_schema)


@pytest.mark.parametrize('table', ['messages','attachments','users','topics'])
@pytest.mark.parametrize('direction', ['child','parent'])
def test_any_participating_inheritance_edge_is_refused(legacy_schema, migration, table, direction):
    if direction=='child':
        legacy_schema.execute(sql.SQL('CREATE TABLE inherited_child () INHERITS ({})').format(sql.Identifier(table)))
    else:
        legacy_schema.execute('CREATE TABLE inherited_parent ()')
        legacy_schema.execute(sql.SQL('ALTER TABLE {} INHERIT inherited_parent').format(sql.Identifier(table)))
    with pytest.raises(ValueError, match='inheritance'):
        migration.migrate_owner_keys(legacy_schema)


def test_failure_after_ddl_rolls_back_all_changes(legacy_schema, migration, monkeypatch):
    def fail(_conn):
        raise RuntimeError('injected verification failure')
    monkeypatch.setattr(migration, '_verify_target', fail)
    with pytest.raises(RuntimeError, match='injected'):
        migration.migrate_owner_keys(legacy_schema)
    assert legacy_schema.execute("SELECT 1 FROM pg_attribute WHERE attrelid='messages'::regclass AND attname='search_id'").fetchone() is None
    assert legacy_schema.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid='messages'::regclass AND contype='p'").fetchone() == ('PRIMARY KEY (id)',)


def test_existing_topic_fk_column_order_is_preserved(legacy_schema, migration):
    conn = legacy_schema
    conn.execute('ALTER TABLE message_topics DROP CONSTRAINT message_topics_user_id_topic_id_fkey')
    conn.execute('ALTER TABLE message_topics ADD FOREIGN KEY(topic_id,user_id) REFERENCES topics(topic_id,user_id)')
    assert migration.migrate_owner_keys(conn)
    assert not migration.migrate_owner_keys(conn)


def test_missing_owner_fk_is_refused(legacy_schema, migration):
    legacy_schema.execute('ALTER TABLE messages DROP CONSTRAINT messages_user_id_fkey')
    with pytest.raises(ValueError, match='dependency'):
        migration.migrate_owner_keys(legacy_schema)


def test_partial_migration_is_refused(legacy_schema, migration):
    legacy_schema.execute('ALTER TABLE messages ADD COLUMN search_id bigint')
    with pytest.raises(ValueError, match='partially migrated'):
        migration.migrate_owner_keys(legacy_schema)


def test_existing_transaction_is_not_committed(legacy_schema, migration):
    with legacy_schema.transaction():
        with pytest.raises(ValueError, match='existing transaction'):
            migration.migrate_owner_keys(legacy_schema)


def test_nonadministrator_role_is_refused_before_ddl(legacy_schema, migration):
    conn = legacy_schema
    role = sql.Identifier('gms_owner_migration_test_' + secrets.token_hex(8))
    conn.execute(sql.SQL('CREATE ROLE {} NOLOGIN').format(role))
    try:
        conn.execute(sql.SQL('SET ROLE {}').format(role))
        with pytest.raises(ValueError, match='direct administrator'):
            migration.migrate_owner_keys(conn)
    finally:
        conn.execute('RESET ROLE')
        conn.execute(sql.SQL('DROP ROLE {}').format(role))


def test_competing_migration_is_refused(legacy_schema, migration):
    with psycopg.connect(make_conninfo(os.environ['GMS_GATEWAY_TEST_DSN'], dbname=legacy_schema.info.dbname)) as other:
        other.execute('SELECT pg_advisory_xact_lock(72341629,1)')
        with pytest.raises(ValueError, match='already running'):
            migration.migrate_owner_keys(legacy_schema)
    assert migration.migrate_owner_keys(legacy_schema)


def test_busy_writer_times_out_without_schema_changes(legacy_schema, migration):
    with psycopg.connect(make_conninfo(os.environ['GMS_GATEWAY_TEST_DSN'], dbname=legacy_schema.info.dbname)) as writer:
        writer.execute("UPDATE messages SET body_text='writer owns transaction'")
        with pytest.raises(psycopg.errors.LockNotAvailable):
            migration.migrate_owner_keys(legacy_schema)
    assert legacy_schema.execute("SELECT 1 FROM pg_attribute WHERE attrelid='messages'::regclass AND attname='search_id'").fetchone() is None
    assert migration.migrate_owner_keys(legacy_schema)


def test_existing_column_grants_and_summary_cascade_are_preserved(legacy_schema, migration):
    conn = legacy_schema
    conn.execute('GRANT SELECT(body_text) ON messages TO PUBLIC')
    migration.migrate_owner_keys(conn)
    body_acl, search_acl = conn.execute("SELECT attname,attacl FROM pg_attribute WHERE attrelid='messages'::regclass AND attname IN ('body_text','search_id') ORDER BY attname").fetchall()
    assert body_acl[0]=='body_text' and body_acl[1]
    assert search_acl == ('search_id',None)
    conn.execute("INSERT INTO messages(user_id,id) VALUES ('bob','cascade')")
    conn.execute("INSERT INTO summary_failures VALUES ('cascade','bob','failure')")
    conn.execute("DELETE FROM messages WHERE user_id='bob' AND id='cascade'")
    assert conn.execute("SELECT 1 FROM summary_failures WHERE user_id='bob'").fetchone() is None


def _real_bm25(conn):
    if not conn.execute("SELECT 1 FROM pg_available_extensions WHERE name='pg_search'").fetchone():
        pytest.skip('Requires real ParadeDB')
    conn.execute('CREATE EXTENSION IF NOT EXISTS pg_search')
    conn.execute('''CREATE INDEX custom_search_index ON messages USING bm25(id,body_text)
        WITH (key_field='id', text_fields='{"body_text":{"tokenizer":{"type":"default"}}}')''')
    conn.execute("COMMENT ON INDEX custom_search_index IS 'synthetic search configuration'")


def test_bm25_preview_is_read_only_and_rebuild_preserves_configuration(legacy_schema, migration):
    conn = legacy_schema
    _real_bm25(conn)
    before = conn.execute("SELECT reloptions FROM pg_class WHERE oid='custom_search_index'::regclass").fetchone()[0]
    plan = migration.preview_owner_keys(conn)
    assert plan['requires_bm25_rebuild'] is True
    assert conn.execute("SELECT 1 FROM pg_attribute WHERE attrelid='messages'::regclass AND attname='search_id'").fetchone() is None
    assert migration.migrate_owner_keys(conn, rebuild_bm25=True)
    after = conn.execute("SELECT reloptions,obj_description(oid) FROM pg_class WHERE oid='custom_search_index'::regclass").fetchone()
    assert set(after[0]) == {option if not option.startswith('key_field=') else 'key_field=search_id' for option in before}
    assert after[1] == 'synthetic search configuration'
    conn.execute("INSERT INTO messages(id,user_id,body_text) VALUES ('same','bob','distinctive bob phrase')")
    rows = conn.execute("SELECT user_id,id,paradedb.score(search_id) FROM messages WHERE messages @@@ 'body_text:original' ORDER BY search_id").fetchall()
    assert len(rows) == 1 and rows[0][:2] == ('alice','same') and rows[0][2] > 0
    assert conn.execute("SELECT user_id FROM messages WHERE messages @@@ 'body_text:distinctive'").fetchall() == [('bob',)]
    conn.execute("UPDATE messages SET body_text='updated bob phrase' WHERE user_id='bob'")
    assert conn.execute("SELECT user_id FROM messages WHERE messages @@@ 'body_text:updated'").fetchall() == [('bob',)]
    assert not migration.migrate_owner_keys(conn, rebuild_bm25=True)


def test_bm25_rebuild_failure_restores_original_index_and_schema(legacy_schema, migration, monkeypatch):
    conn = legacy_schema
    _real_bm25(conn)
    definition = conn.execute("SELECT pg_get_indexdef('custom_search_index'::regclass)").fetchone()
    def fail(_conn):
        raise RuntimeError('injected after BM25 rebuild')
    monkeypatch.setattr(migration, '_verify_target', fail)
    with pytest.raises(RuntimeError, match='injected'):
        migration.migrate_owner_keys(conn, rebuild_bm25=True)
    assert conn.execute("SELECT pg_get_indexdef('custom_search_index'::regclass)").fetchone() == definition
    assert conn.execute("SELECT 1 FROM pg_attribute WHERE attrelid='messages'::regclass AND attname='search_id'").fetchone() is None
    assert conn.execute("SELECT id FROM messages WHERE messages @@@ 'body_text:original'").fetchall() == [('same',)]


def test_fresh_schema_has_owner_keys_and_real_bm25_and_reruns(legacy_schema, migration):
    conn = legacy_schema
    if not conn.execute("SELECT 1 FROM pg_available_extensions WHERE name='pg_search'").fetchone():
        pytest.skip('Requires real ParadeDB')
    conn.execute('DROP SCHEMA public CASCADE; CREATE SCHEMA public')
    schema = (Path(__file__).parents[1] / 'src/gmail_search/store/pg_schema.sql').read_text()
    conn.execute(schema)
    assert migration.migrate_owner_keys(conn, rebuild_bm25=True) is False
    conn.execute(schema)
    conn.execute("INSERT INTO users(id,email) VALUES ('alice','alice@example.test'),('bob','bob@example.test')")
    conn.execute("""INSERT INTO messages(user_id,id,thread_id,from_addr,to_addr,date,body_text)
        VALUES ('alice','same','thread','from','to','2026-01-01','alice original'),
               ('bob','same','thread','from','to','2026-01-01','bob distinct')""")
    assert conn.execute("SELECT user_id,paradedb.score(search_id)>0 FROM messages WHERE messages @@@ 'body_text:distinct'").fetchall() == [('bob',True)]


def test_bm25_preview_rejects_bad_owner_data_without_index_changes(legacy_schema, migration):
    conn = legacy_schema
    _real_bm25(conn)
    conn.execute("UPDATE attachments SET user_id='bob'")
    with pytest.raises(ValueError, match='owner'):
        migration.preview_owner_keys(conn)
    assert conn.execute("SELECT reloptions FROM pg_class WHERE oid='custom_search_index'::regclass").fetchone()[0][0] == 'key_field=id'


def test_bm25_delete_and_reinsert_preserve_other_owner(legacy_schema, migration):
    conn = legacy_schema
    _real_bm25(conn)
    migration.migrate_owner_keys(conn, rebuild_bm25=True)
    conn.execute("INSERT INTO messages(user_id,id,body_text) VALUES ('bob','same','original bob')")
    bob_search_id = conn.execute("SELECT search_id FROM messages WHERE user_id='bob'").fetchone()[0]
    conn.execute("DELETE FROM messages WHERE user_id='bob'")
    assert conn.execute("SELECT user_id FROM messages WHERE messages @@@ 'body_text:original'").fetchall() == [('alice',)]
    conn.execute("INSERT INTO messages(user_id,id,body_text) VALUES ('bob','same','original bob')")
    assert conn.execute("SELECT search_id FROM messages WHERE user_id='bob'").fetchone()[0] != bob_search_id
    assert conn.execute("SELECT user_id FROM messages WHERE messages @@@ 'body_text:original' ORDER BY user_id").fetchall() == [('alice',),('bob',)]


def test_bm25_rebuild_preserves_column_collation(legacy_schema, migration):
    conn = legacy_schema
    if not conn.execute("SELECT 1 FROM pg_available_extensions WHERE name='pg_search'").fetchone():
        pytest.skip('Requires real ParadeDB')
    conn.execute('CREATE EXTENSION IF NOT EXISTS pg_search')
    conn.execute('''CREATE INDEX custom_search_index ON messages USING bm25(id,body_text COLLATE "C") WITH (key_field='id')''')
    migration.migrate_owner_keys(conn, rebuild_bm25=True)
    assert conn.execute("SELECT indcollation[2] FROM pg_index WHERE indexrelid='custom_search_index'::regclass").fetchone() == conn.execute("SELECT oid FROM pg_collation WHERE collname='C' AND collnamespace='pg_catalog'::regnamespace").fetchone()


@pytest.mark.parametrize('candidate_ids', [None, ['same']])
def test_application_bm25_keeps_colliding_owners_separate(legacy_schema, migration, candidate_ids):
    import logging
    from psycopg.rows import dict_row
    from gmail_search.store.queries import _pg_bm25_messages
    conn = legacy_schema
    _real_bm25(conn)
    migration.migrate_owner_keys(conn, rebuild_bm25=True)
    conn.execute("INSERT INTO messages(user_id,id,body_text) VALUES ('bob','same','distinctive bob')")
    conn.row_factory = dict_row
    logger = logging.getLogger(__name__)
    alice = _pg_bm25_messages(conn, 'body_text:original', 10, logger, candidate_ids, user_id='alice')
    assert set(alice) == {'same'} and alice['same'] > 0
    assert _pg_bm25_messages(conn, 'body_text:original', 10, logger, candidate_ids, user_id='bob') == {}
    bob = _pg_bm25_messages(conn, 'body_text:distinctive', 10, logger, candidate_ids, user_id='bob')
    assert set(bob) == {'same'} and bob['same'] > 0


def test_startup_schema_does_not_promote_existing_legacy_keys(legacy_schema, migration):
    conn = legacy_schema
    if not conn.execute("SELECT 1 FROM pg_available_extensions WHERE name='pg_search'").fetchone():
        pytest.skip('Requires real ParadeDB')
    conn.execute('DROP SCHEMA public CASCADE; CREATE SCHEMA public')
    schema = (Path(__file__).parents[1] / 'src/gmail_search/store/pg_schema.sql').read_text()
    conn.execute(schema)
    # Build a complete legacy installation with all the ordinary startup columns.
    # This reverse setup is safe only because this disposable schema is empty.
    conn.execute('DROP INDEX messages_bm25_idx')
    constraints = migration._constraints(conn)
    for c in constraints:
        if c['kind']=='f' and c['remote_table'] in ('messages','attachments'):
            conn.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT {}').format(sql.Identifier(c['table']),sql.Identifier(c['name'])))
    for c in constraints:
        if c['kind'] in ('p','u') and c['table']!='embeddings' and not (c['table']=='attachments' and c['kind']=='p'):
            conn.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT {}').format(sql.Identifier(c['table']),sql.Identifier(c['name'])))
    conn.execute('ALTER TABLE messages DROP COLUMN search_id')
    for table in migration.TABLES:
        for kind, columns in migration.LEGACY_KEYS[table] - migration.TARGET_KEYS[table]:
            conn.execute(sql.SQL('ALTER TABLE {} ADD {} ({})').format(sql.Identifier(table),sql.SQL('PRIMARY KEY' if kind=='p' else 'UNIQUE'),sql.SQL(',').join(map(sql.Identifier,columns))))
    for table, columns, parent, remote, delete in migration._foreign_keys(False):
        if parent=='topics':
            continue
        conn.execute(sql.SQL('ALTER TABLE {} ADD FOREIGN KEY ({}) REFERENCES {} ({}) ON DELETE {}').format(sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,columns)),sql.Identifier(parent),sql.SQL(',').join(map(sql.Identifier,remote)),sql.SQL('CASCADE' if delete=='c' else 'NO ACTION')))
    conn.execute(schema)
    assert migration.preview_owner_keys(conn)['target_schema'] is False
    assert conn.execute("SELECT 1 FROM pg_attribute WHERE attrelid='messages'::regclass AND attname='search_id' AND NOT attisdropped").fetchone() is None
    assert conn.execute("SELECT reloptions FROM pg_class WHERE oid='messages_bm25_idx'::regclass").fetchone() == (['key_field=id'],)
