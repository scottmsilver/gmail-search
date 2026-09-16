"""Reader ACL/RLS regression on real partitioned parents, without search extensions.

This qualifies the existing immutable reader before the migration is integrated;
it does not stand in for BM25 ranking or migration tests.
"""
import secrets

import psycopg
from psycopg import sql
import pytest

from gmail_search.gateway.database import reader_role
from gmail_search.gateway.provision import provision_reader
from test_gateway_database_integration import database as database_fixture, reader_dsn


database = database_fixture


@pytest.fixture
def partitioned(database):
    dsn, owners = database
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.transaction():
            conn.execute('ALTER TABLE public.messages RENAME TO old_messages')
            conn.execute('CREATE TABLE public.messages (LIKE public.old_messages INCLUDING DEFAULTS) PARTITION BY LIST(user_id)')
            conn.execute('ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY')
            conn.execute('ALTER TABLE public.messages FORCE ROW LEVEL SECURITY')
            conn.execute('CREATE POLICY legacy ON public.messages USING (true)')
            for ordinal, owner in enumerate(owners):
                child = sql.Identifier('public', 'mail_partition_' + str(ordinal))
                conn.execute(sql.SQL('CREATE TABLE {} PARTITION OF public.messages FOR VALUES IN ({})').format(child, sql.Literal(owner)))
                conn.execute(sql.SQL('ALTER TABLE {} ENABLE ROW LEVEL SECURITY').format(child))
                conn.execute(sql.SQL('ALTER TABLE {} FORCE ROW LEVEL SECURITY').format(child))
            conn.execute('INSERT INTO public.messages SELECT * FROM public.old_messages')
            conn.execute('DROP TABLE public.old_messages')
            for owner in owners:
                provision_reader(conn, owner, secrets.token_urlsafe(40))
    return dsn, owners


def test_parent_rls_fixes_owner_and_direct_children_are_denied(partitioned):
    dsn, owners = partitioned
    for owner, foreign in (owners, owners[::-1]):
        with psycopg.connect(reader_dsn(dsn, owner), autocommit=True) as conn:
            conn.execute("SELECT set_config('app.user_id', %s, false)", (foreign,))
            assert conn.execute('SELECT user_id, subject FROM public.messages').fetchall() == [(owner, owner + ' private')]
            assert conn.execute('SELECT subject FROM public.messages WHERE user_id=%s', (foreign,)).fetchall() == []
            for child in ('mail_partition_0', 'mail_partition_1'):
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    conn.execute(sql.SQL('SELECT subject FROM public.{}').format(sql.Identifier(child)))
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute('SELECT raw_json FROM public.messages')


@pytest.mark.parametrize('grantee', ['reader', 'public'])
@pytest.mark.parametrize('privilege', ['SELECT(subject)', 'UPDATE(subject)', 'DELETE'])
def test_provisioning_rejects_effective_child_grants(partitioned, grantee, privilege):
    dsn, (alice, _) = partitioned
    target = sql.Identifier(reader_role(alice)) if grantee == 'reader' else sql.SQL('PUBLIC')
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL('GRANT {} ON public.mail_partition_1 TO {}').format(sql.SQL(privilege), target))
        with pytest.raises(ValueError, match='Unexpected effective'):
            provision_reader(conn, alice, secrets.token_urlsafe(40))
