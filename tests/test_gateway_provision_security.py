"""Adversarial reader qualification against disposable synthetic databases only."""
from pathlib import Path
import secrets

import psycopg
from psycopg import sql
import pytest

ROOT = Path(__file__).parents[1]

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


# ── the extension-metadata allowlist ─────────────────────────────────────────

def _public_provisioner():
    """`deploy/public/provision_database.py` is a standalone admin script with no
    package import, so it cannot share a constant by importing one. Load it by
    path instead and compare, rather than letting two copies drift apart."""
    import importlib.util

    path = ROOT / 'deploy/public/provision_database.py'
    spec = importlib.util.spec_from_file_location('provision_database', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extension_allowlist_matches_the_public_provisioner():
    """Two provisioning paths, one judgement. They disagreed before: the public
    login allowlisted this metadata as harmless while `verify_reader_access`
    rejected it, so a per-owner reader could not be provisioned at all against
    the live database."""
    from gmail_search.gateway.schema import EXTENSION_METADATA

    assert set(EXTENSION_METADATA) == set(_public_provisioner().EXTENSION_METADATA)


def test_the_allowlist_can_never_cover_mail():
    """The allowlist exists for extension catalogues. If a mail relation ever
    appears in it, a per-owner reader silently gains cross-owner read access —
    which is the whole thing readers exist to prevent."""
    from gmail_search.gateway.schema import ANALYTICAL_SCHEMA, EXTENSION_METADATA

    mail_schemas = {'public'}
    for schema, table in EXTENSION_METADATA:
        assert table not in ANALYTICAL_SCHEMA, f'{schema}.{table} is a mail relation'
        if schema in mail_schemas:
            # Only PostGIS's own views live in `public`; nothing else may.
            assert table in {'geography_columns', 'geometry_columns', 'spatial_ref_sys'}, \
                f'unexpected public relation allowlisted: {table}'


def test_the_allowlist_excludes_the_view_definition_catalogue():
    """`pgivm.pg_ivm_immv.viewdef` stores the SQL of any incremental materialised
    view. It is empty today, but an IMMV built over mail would put mail column
    and predicate text in a PUBLIC-readable catalogue. It is deliberately not
    allowlisted; PUBLIC access to it is revoked instead."""
    from gmail_search.gateway.schema import EXTENSION_METADATA

    assert ('pgivm', 'pg_ivm_immv') not in EXTENSION_METADATA


def test_search_reader_allowlist_is_the_same_one():
    """`verify_search_reader_access` had no allowlist at all and rejected the
    same extension catalogues, so provisioning stopped a second time after the
    reader role already existed. One judgement, applied in both verifiers."""
    from gmail_search.gateway import provision_search_reader
    from gmail_search.gateway.schema import EXTENSION_METADATA

    assert provision_search_reader.EXTENSION_METADATA is EXTENSION_METADATA


# ── search_path determinism ──────────────────────────────────────────────────

class _Recorder:
    """A connection that records statements and stops the check sequence early."""

    class _Stop(Exception):
        pass

    def __init__(self, stop_after):
        self.statements, self.stop_after = [], stop_after

    def execute(self, statement, params=None):
        self.statements.append(statement)
        # Raise exactly once, so the restore in the `finally` still records.
        if len(self.statements) == self.stop_after + 1:
            raise self._Stop()
        return self

    def fetchall(self):
        return [('public, paradedb',)]


def test_search_verification_pins_search_path_and_restores_it():
    """`regtype::text` omits a type's schema when that type is visible on the
    caller's search_path. The pinned pg_search signatures therefore resolved
    differently per caller: against this database, whose session default is
    `public, paradedb`, the administrator saw

        'searchqueryinput'  where the pin expects  'paradedb.searchqueryinput'

    so provisioning could never succeed for the one role that performs it, while
    passing in tests whose database has no `paradedb` on the path. Verification
    must therefore fix the path itself rather than inherit one.
    """
    from gmail_search.gateway import provision_search_reader as module

    recorder = _Recorder(stop_after=2)
    with pytest.raises(_Recorder._Stop):
        module.verify_search_reader_access(recorder, 'u_owner')

    joined = ' | '.join(str(s) for s in recorder.statements)
    assert 'search_path' in recorder.statements[0], 'must read the caller path first'
    assert any('pg_catalog' in str(s) for s in recorder.statements[:2]), \
        f'must pin the path before any pinned lookup: {joined}'


def test_search_verification_restores_the_caller_path_on_failure():
    """A refused provision must not leave the administrator connection with a
    rewritten search_path; later statements in the same session would resolve
    against the wrong schemas."""
    from gmail_search.gateway import provision_search_reader as module

    recorder = _Recorder(stop_after=2)
    with pytest.raises(_Recorder._Stop):
        module.verify_search_reader_access(recorder, 'u_owner')
    assert any('set_config' in str(s) or 'search_path' in str(s)
               for s in recorder.statements[-1:]), 'the path must be restored even on failure'


def test_writer_shares_the_same_allowlist():
    """The fourth verifier with the same defect. `provision_application_writer`
    rejected `pdb.index_layer_info` exactly as the reader and search verifiers
    did, for the same reason and with the same remedy."""
    from gmail_search.gateway import provision_writer
    from gmail_search.gateway.schema import EXTENSION_METADATA

    assert provision_writer.EXTENSION_METADATA is EXTENSION_METADATA
