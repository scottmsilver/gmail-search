"""Adversarial reader qualification against disposable synthetic databases only."""
import secrets

import psycopg
from psycopg import sql
import pytest

from test_gateway_database_integration import database as database_fixture, reader_dsn
from gmail_search.gateway.database import reader_role
from gmail_search.gateway.provision import provision_reader

database = database_fixture
from gmail_search.gateway.schema import ANALYTICAL_SCHEMA


def test_every_table_keeps_fixed_owner_when_session_identity_changes(database):
    dsn, owners = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        for table in ANALYTICAL_SCHEMA:
            if table != 'messages':
                for owner in owners:
                    admin.execute(sql.SQL('INSERT INTO public.{} (user_id) VALUES (%s)').format(sql.Identifier(table)), (owner,))
    for owner, foreign in (owners, owners[::-1]):
        with psycopg.connect(reader_dsn(dsn, owner), autocommit=True) as reader:
            reader.execute("SELECT set_config('app.user_id', %s, false)", (foreign,))
            for table in ANALYTICAL_SCHEMA:
                assert reader.execute(sql.SQL('SELECT user_id FROM public.{}').format(sql.Identifier(table))).fetchall() == [(owner,)]
                assert reader.execute(sql.SQL('SELECT user_id FROM public.{} WHERE user_id=%s').format(sql.Identifier(table)), (foreign,)).fetchall() == []


def test_acls_deny_writes_even_when_reader_disables_default_read_only(database):
    dsn, (alice, _) = database
    with psycopg.connect(reader_dsn(dsn, alice), autocommit=True) as reader:
        reader.execute('SET default_transaction_read_only=off')
        assert reader.execute('SHOW transaction_read_only').fetchone() == ('off',)
        statements = [
            'CREATE TABLE public.evil (id int)', 'CREATE TEMP TABLE evil (id int)',
            'SELECT raw_json FROM public.messages', 'SELECT * FROM public.users',
        ]
        for table in ANALYTICAL_SCHEMA:
            relation = sql.Identifier('public', table)
            statements.extend([
                sql.SQL('INSERT INTO {} (user_id) VALUES ({})').format(relation, sql.Literal(alice)),
                sql.SQL('UPDATE {} SET user_id={}').format(relation, sql.Literal(alice)),
                sql.SQL('DELETE FROM {}').format(relation),
                sql.SQL('TRUNCATE {}').format(relation),
            ])
        for statement in statements:
            with pytest.raises(psycopg.errors.InsufficientPrivilege) as failure:
                reader.execute(statement)
            assert failure.value.sqlstate == '42501'


@pytest.mark.parametrize('reverse', [False, True])
def test_reader_role_cannot_have_memberships_or_members(database, reverse):
    dsn, (alice, bob) = database
    grant_role, member = (alice, bob) if reverse else (bob, alice)
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL('GRANT {} TO {}').format(sql.Identifier(reader_role(grant_role)), sql.Identifier(reader_role(member))))
        with pytest.raises(ValueError, match='memberships or members'):
            provision_reader(admin, alice, secrets.token_urlsafe(40))


@pytest.mark.parametrize('attribute', ['SUPERUSER', 'CREATEDB', 'CREATEROLE', 'REPLICATION', 'BYPASSRLS'])
def test_privileged_existing_reader_is_rejected_without_silent_repair(database, attribute):
    dsn, (alice, _) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        role = sql.Identifier(reader_role(alice))
        admin.execute(sql.SQL('ALTER ROLE {} {}').format(role, sql.SQL(attribute)))
        try:
            with pytest.raises(ValueError, match='privileged attributes'):
                provision_reader(admin, alice, secrets.token_urlsafe(40))
            column = {'SUPERUSER':'rolsuper', 'CREATEDB':'rolcreatedb', 'CREATEROLE':'rolcreaterole', 'REPLICATION':'rolreplication', 'BYPASSRLS':'rolbypassrls'}[attribute]
            assert admin.execute(sql.SQL('SELECT {} FROM pg_roles WHERE rolname=%s').format(sql.Identifier(column)), (reader_role(alice),)).fetchone() == (True,)
        finally:
            admin.execute(sql.SQL('ALTER ROLE {} {}').format(role, sql.SQL('NO' + attribute)))


def test_reader_owning_objects_is_rejected(database):
    dsn, (alice, _) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute('CREATE TABLE public.reader_owned (id int)')
        admin.execute(sql.SQL('ALTER TABLE public.reader_owned OWNER TO {}').format(sql.Identifier(reader_role(alice))))
        with pytest.raises(ValueError, match='own database objects'):
            provision_reader(admin, alice, secrets.token_urlsafe(40))


@pytest.mark.parametrize('mutation', ['no_rls', 'native_type', 'domain_type'])
def test_analytical_schema_changes_fail_closed(database, mutation):
    dsn, (alice, _) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        if mutation == 'no_rls':
            admin.execute('ALTER TABLE public.messages DISABLE ROW LEVEL SECURITY')
        elif mutation == 'native_type':
            admin.execute('ALTER TABLE public.messages ALTER COLUMN subject TYPE varchar')
        else:
            admin.execute('CREATE DOMAIN public.mail_text AS text')
            admin.execute('ALTER TABLE public.messages ALTER COLUMN subject TYPE public.mail_text')
        with pytest.raises(ValueError, match='RLS enabled|column/type mismatch'):
            provision_reader(admin, alice, secrets.token_urlsafe(40))


@pytest.mark.parametrize('grant', ['TEMP', 'CREATE_DATABASE', 'CREATE_SCHEMA', 'SELECT_COLUMN', 'UPDATE_COLUMN'])
def test_effective_public_grants_fail_closed(database, grant):
    dsn, (alice, _) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        if grant in ('TEMP', 'CREATE_DATABASE'):
            privilege = 'TEMP' if grant == 'TEMP' else 'CREATE'
            admin.execute(sql.SQL('GRANT {} ON DATABASE {} TO PUBLIC').format(sql.SQL(privilege), sql.Identifier(admin.info.dbname)))
        elif grant == 'CREATE_SCHEMA':
            admin.execute('GRANT CREATE ON SCHEMA public TO PUBLIC')
        elif grant == 'SELECT_COLUMN':
            admin.execute('GRANT SELECT (raw_json) ON public.messages TO PUBLIC')
        else:
            admin.execute('GRANT UPDATE (subject) ON public.messages TO PUBLIC')
        with pytest.raises(ValueError, match='grant|CREATE/TEMP'):
            provision_reader(admin, alice, secrets.token_urlsafe(40))
