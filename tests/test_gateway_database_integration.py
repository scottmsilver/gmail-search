"""Opt-in synthetic PostgreSQL isolation tests. Never point at a real mailbox DB.

GMS_GATEWAY_TEST_DSN must identify an empty disposable test database. Tests
create their own database so no existing schema is modified.
"""
import os
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from gmail_search.gateway.database import reader_role
from gmail_search.gateway.provision import provision_reader
from gmail_search.gateway.schema import ANALYTICAL_SCHEMA


@pytest.fixture
def database():
    dsn = os.environ.get('GMS_GATEWAY_TEST_DSN')
    if not dsn:
        pytest.skip('Set GMS_GATEWAY_TEST_DSN for isolated synthetic PostgreSQL tests')
    suffix = secrets.token_hex(8)
    name = 'gms_gateway_test_' + suffix
    owners = ('alice_' + suffix, 'bob_' + suffix)
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    cfg = conninfo_to_dict(dsn)
    cfg['dbname'] = name
    target = make_conninfo(**cfg)
    try:
        with psycopg.connect(target, autocommit=True) as conn:
            conn.execute(sql.SQL('REVOKE TEMP ON DATABASE {} FROM PUBLIC').format(sql.Identifier(name)))
            conn.execute('CREATE TABLE public.users (id text PRIMARY KEY)')
            for owner in owners:
                conn.execute('INSERT INTO public.users VALUES (%s)', (owner,))
            for table, columns in ANALYTICAL_SCHEMA.items():
                defs = sql.SQL(', ').join(sql.SQL('{} {}').format(sql.Identifier(col), sql.SQL(kind)) for col, kind in columns.items())
                conn.execute(sql.SQL('CREATE TABLE public.{} ({})').format(sql.Identifier(table), defs))
                conn.execute(sql.SQL('ALTER TABLE public.{} ENABLE ROW LEVEL SECURITY').format(sql.Identifier(table)))
                # Simulates an existing broad legacy policy; restrictive reader policy must win.
                conn.execute(sql.SQL('CREATE POLICY legacy ON public.{} USING (true)').format(sql.Identifier(table)))
            for owner in owners:
                conn.execute('INSERT INTO public.messages (id, subject, user_id) VALUES (%s, %s, %s)', (owner, owner + ' private', owner))
                provision_reader(conn, owner, secrets.token_urlsafe(40))
            conn.execute("ALTER TABLE public.messages ADD COLUMN raw_json text DEFAULT 'withheld'")
        yield target, owners
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))
            for owner in owners:
                admin.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(reader_role(owner))))


def reader_dsn(dsn, owner):
    cfg = conninfo_to_dict(dsn)
    cfg['user'] = reader_role(owner)
    cfg.pop('password', None)
    return make_conninfo(**cfg)


def test_unfiltered_select_and_identity_setting_cannot_cross_users(database):
    dsn, (alice, bob) = database
    for owner, foreign in ((alice, bob), (bob, alice)):
        with psycopg.connect(reader_dsn(dsn, owner), autocommit=True) as conn:
            assert conn.execute('SELECT user_id FROM public.messages').fetchall() == [(owner,)]
            conn.execute("SELECT set_config('app.user_id', %s, false)", (foreign,))
            assert conn.execute('SELECT user_id FROM public.messages').fetchall() == [(owner,)]
            assert conn.execute('SELECT subject FROM public.messages WHERE user_id=%s', (foreign,)).fetchall() == []
            assert conn.execute('SELECT session_user, current_user').fetchone() == (reader_role(owner), reader_role(owner))
            for statement in (
                sql.SQL('SET ROLE {}').format(sql.Identifier(reader_role(foreign))),
                'SET ROLE postgres',
                'UPDATE public.messages SET subject=\'changed\'',
                'DELETE FROM public.messages',
                'CREATE TABLE public.evil (id int)',
                'CREATE TEMP TABLE evil (id int)',
                'SELECT * FROM public.users',
                'SELECT raw_json FROM public.messages',
            ):
                with pytest.raises(psycopg.Error):
                    conn.execute(statement)
            conn.execute('SET row_security=off')
            with pytest.raises(psycopg.Error):
                conn.execute('SELECT subject FROM public.messages')


def test_reprovision_refuses_owner_mismatch_and_privileged_membership(database):
    dsn, (alice, bob) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        role = sql.Identifier(reader_role(alice))
        admin.execute(sql.SQL('COMMENT ON ROLE {} IS {}').format(role, sql.Literal('wrong-owner')))
        with pytest.raises(ValueError, match='binding'):
            provision_reader(admin, alice, secrets.token_urlsafe(40))
        # No partial reset to a more permissive role on failure.
        assert admin.execute('SELECT rolcanlogin FROM pg_roles WHERE rolname=%s', (reader_role(alice),)).fetchone()[0]


def test_public_function_and_table_grants_prevent_provisioning(database):
    dsn, (alice, bob) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute('CREATE TABLE public.private_global (secret text)')
        admin.execute('GRANT SELECT ON public.private_global TO PUBLIC')
        with pytest.raises(ValueError, match='grant'):
            provision_reader(admin, alice, secrets.token_urlsafe(40))
        admin.execute('REVOKE SELECT ON public.private_global FROM PUBLIC')
        admin.execute("CREATE FUNCTION public.leak() RETURNS text LANGUAGE sql SECURITY DEFINER AS 'SELECT secret FROM public.private_global LIMIT 1'")
        with pytest.raises(ValueError, match='SECURITY DEFINER'):
            provision_reader(admin, alice, secrets.token_urlsafe(40))


def test_public_sequence_grants_prevent_provisioning(database):
    dsn, (alice, bob) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute('CREATE SEQUENCE public.internal_sequence')
        admin.execute('GRANT USAGE ON SEQUENCE public.internal_sequence TO PUBLIC')
        with pytest.raises(ValueError, match='sequence'):
            provision_reader(admin, alice, secrets.token_urlsafe(40))


def test_compiler_queries_execute_on_real_postgres(database):
    from decimal import Decimal
    from gmail_search.gateway.analytics import compile_query
    dsn, (alice, bob) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute('INSERT INTO public.attachments (id,message_id,size_bytes,user_id) VALUES (1,%s,17,%s),(2,%s,21,%s)', (alice, alice, bob, bob))
    with psycopg.connect(reader_dsn(dsn, alice), autocommit=True) as conn:
        cases = [
            ('SELECT count(*) AS total FROM messages', [(1,)]),
            ('SELECT 1.5 AS value', [(Decimal('1.5'),)]),
            ('SELECT -size_bytes AS negative FROM attachments', [(-17,)]),
            ('SELECT size_bytes % 10 AS tail FROM attachments', [(7,)]),
            ('SELECT id FROM messages WHERE id IN (SELECT message_id FROM attachments)', [(alice,)]),
            ('SELECT m.id, (SELECT count(*) FROM attachments a WHERE a.message_id=m.id) AS n FROM messages m', [(alice, 1)]),
            ('SELECT row_number() OVER (ORDER BY id) AS rank FROM messages', [(1,)]),
            ('WITH a AS (SELECT message_id, sum(size_bytes) AS bytes FROM attachments GROUP BY message_id) SELECT m.id,a.bytes FROM messages m JOIN a ON a.message_id=m.id', [(alice, Decimal('17'))]),
            ('SELECT coalesce(NULL, size_bytes) AS n FROM attachments', [(17,)]),
        ]
        for query, expected in cases:
            compiled = compile_query(query)
            assert conn.execute(compiled.sql, compiled.params).fetchall() == expected
