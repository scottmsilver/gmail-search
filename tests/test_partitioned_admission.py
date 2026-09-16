"""Invited admission publishes credentials only after real owner partitions exist."""
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from gmail_search.auth.identity_store import Account, VerifiedGoogleIdentity
from gmail_search.gateway.admission_provision import AdmissionProvisioner
from gmail_search.gateway.database import reader_role
from gmail_search.gateway.partition_profiles import NUMERIC_OWNER_PARTITIONS_V1 as NUMERIC
from gmail_search.gateway.writer import application_writer_role
from gmail_search.gateway.schema import ANALYTICAL_SCHEMA
from gmail_search.gateway.provision_writer import _COLUMNS, _SEQUENCES
from test_gateway_partition_provision import partition_database as partition_database_fixture

partition_database = partition_database_fixture


@pytest.fixture
def admission_database(partition_database):
    owner = 'admit_' + secrets.token_hex(10)
    roles = [reader_role(owner), application_writer_role(owner)]
    with psycopg.connect(partition_database, autocommit=True) as admin:
        admin.execute(sql.SQL('REVOKE TEMP ON DATABASE {} FROM PUBLIC').format(sql.Identifier(admin.info.dbname)))
        # Revoke extension SECURITY DEFINER defaults in this disposable DB only.
        for (routine,) in admin.execute("SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE p.prosecdef AND n.nspname NOT IN ('pg_catalog','information_schema')"):
            admin.execute(sql.SQL('REVOKE ALL ON FUNCTION {} FROM PUBLIC').format(sql.SQL(routine)))
        admin.execute('REVOKE CREATE ON SCHEMA public,paradedb,pdb FROM PUBLIC')
        admin.execute('REVOKE ALL ON ALL TABLES IN SCHEMA public,paradedb,pdb FROM PUBLIC')
        admin.execute('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public,paradedb,pdb FROM PUBLIC')
        admin.execute('ALTER TABLE public.users ADD COLUMN email text UNIQUE, ADD COLUMN google_sub text UNIQUE')
        for table, columns in _COLUMNS.items():
            if table == 'browser_answer_receipts':
                continue
            definitions = []
            for column, kind in columns.items():
                if column == 'id' and table in _SEQUENCES:
                    definition = sql.SQL('{} bigserial PRIMARY KEY').format(sql.Identifier(column))
                else:
                    definition = sql.SQL('{} {}').format(sql.Identifier(column),sql.SQL(kind))
                    if column == 'id':
                        definition += sql.SQL(' PRIMARY KEY')
                definitions.append(definition)
            admin.execute(sql.SQL('CREATE TABLE public.{} ({})').format(sql.Identifier(table), sql.SQL(',').join(definitions)))
            admin.execute(sql.SQL('ALTER TABLE public.{} ENABLE ROW LEVEL SECURITY').format(sql.Identifier(table)))
        from gmail_search.gateway.browser_conversations import BrowserConversations
        BrowserConversations.install(admin)
        for table, columns in ANALYTICAL_SCHEMA.items():
            if table not in _COLUMNS and table not in ('messages','attachments','propositions'):
                admin.execute(sql.SQL('CREATE TABLE public.{} ({})').format(sql.Identifier(table),sql.SQL(',').join(
                    sql.SQL('{} {}').format(sql.Identifier(col),sql.SQL(kind)) for col,kind in columns.items())))
            else:
                for col, kind in columns.items():
                    admin.execute(sql.SQL('ALTER TABLE public.{} ADD COLUMN IF NOT EXISTS {} {}').format(sql.Identifier(table),sql.Identifier(col),sql.SQL(kind)))
            admin.execute(sql.SQL('ALTER TABLE public.{} ENABLE ROW LEVEL SECURITY').format(sql.Identifier(table)))
        try:
            yield partition_database, owner
        finally:
            for role in roles:
                if admin.execute('SELECT 1 FROM pg_roles WHERE rolname=%s',(role,)).fetchone():
                    admin.execute(sql.SQL('DROP OWNED BY {}').format(sql.Identifier(role)))
                    admin.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))


def admission(dsn, owner, installer):
    config = conninfo_to_dict(dsn)
    config.pop('user',None)
    config.pop('password',None)
    provision = AdmissionProvisioner(lambda: psycopg.connect(dsn), runtime_dsn=make_conninfo(**config), install_credentials=installer, partition_profile=NUMERIC)
    account = Account(owner, owner+'@example.test',1)
    identity = VerifiedGoogleIdentity(account.email, 'google-'+owner, True)
    return lambda: provision(account,identity)


def test_credentials_are_published_after_all_three_partitions_commit(admission_database):
    from gmail_search.gateway.partitions import verify_owner_partitions
    dsn, owner = admission_database
    installed = []
    def install(reader, writer):
        # Separate connection proves readiness is committed before publication.
        with psycopg.connect(dsn,autocommit=True) as admin:
            verify_owner_partitions(admin,owner)
            assert admin.execute('SELECT google_sub FROM public.users WHERE id=%s',(owner,)).fetchone() == ('google-'+owner,)
        installed.append((reader,writer))
    assert admission(dsn,owner,install)() is True
    assert len(installed) == 1
    assert installed[0][0].owner_id == installed[0][1].owner_id == owner
    with psycopg.connect(installed[0][0].dsn,autocommit=True) as reader:
        assert reader.execute('SELECT count(*) FROM public.messages').fetchone() == (0,)


def test_missing_parent_index_rolls_back_account_and_publishes_nothing(admission_database):
    dsn, owner = admission_database
    with psycopg.connect(dsn,autocommit=True) as admin:
        admin.execute('DROP INDEX public.props_bm25_idx')
    installed = []
    with pytest.raises(ValueError):
        admission(dsn,owner,lambda *credentials: installed.append(credentials))()
    assert installed == []
    with psycopg.connect(dsn,autocommit=True) as admin:
        assert admin.execute('SELECT 1 FROM public.users WHERE id=%s',(owner,)).fetchone() is None
        assert admin.execute('SELECT 1 FROM pg_roles WHERE rolname=%s',(reader_role(owner),)).fetchone() is None
