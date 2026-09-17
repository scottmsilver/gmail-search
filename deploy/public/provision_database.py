#!/usr/bin/env python3
"""Provision an owner-pinned public login. SQL preview by default; --apply opts in.

Run with an administrator DSN in GMAIL_PROVISION_DSN (never a CLI argument).
Password is created/reused only with --apply, in an exclusive mode-0600 file.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import stat

ROLE = 'gmail_search_public'
READ_TABLES = (
    'messages', 'attachments', 'embeddings', 'thread_summary', 'message_summaries',
    'propositions', 'prop_processed', 'topics', 'message_topics', 'contact_frequency',
    'term_aliases', 'scann_index_pointer', 'users', 'agent_artifacts',
)
WRITE_GRANTS = {
    'conversations': 'SELECT, INSERT, UPDATE, DELETE',
    'conversation_messages': 'SELECT, INSERT, UPDATE, DELETE',
    'agent_sessions': 'SELECT, INSERT, UPDATE',
    'agent_events': 'SELECT, INSERT',
    'costs': 'SELECT, INSERT',
    # Battles are available to owners the server reports as capable. Both battle
    # endpoints are per-owner scoped; the vote inserts, the leaderboard reads.
    'model_battles': 'SELECT, INSERT',
}
SEQUENCES = ('conversation_messages_id_seq', 'agent_events_id_seq', 'costs_id_seq', 'model_battles_id_seq')
ALL_TABLES = (*READ_TABLES, *WRITE_GRANTS)
# Preexisting PUBLIC read-only extension metadata; none contains mail or queries.
EXTENSION_METADATA = {
    ('pdb', 'index_layer_info'), ('paradedb', 'index_layer_info'),
    ('public', 'geography_columns'), ('public', 'geometry_columns'), ('public', 'spatial_ref_sys'),
    ('topology', 'topology'), ('topology', 'layer'),
    *(('tiger', name) for name in ('geocode_settings', 'geocode_settings_default',
        'loader_platform', 'loader_variables', 'loader_lookuptables', 'pagc_gaz', 'pagc_lex', 'pagc_rules')),
}


def ident(value):
    return '"' + value.replace('"', '""') + '"'


def literal(value):
    return "'" + value.replace("'", "''") + "'"


def owner_predicate(table, owner_id):
    owner = literal(owner_id)
    if table == 'users':
        return f'id::text = {owner}'
    if table == 'conversation_messages':
        return ('EXISTS (SELECT 1 FROM public.conversations c '
                f'WHERE c.id = conversation_messages.conversation_id AND c.user_id::text = {owner})')
    if table in ('agent_events', 'agent_artifacts'):
        return ('EXISTS (SELECT 1 FROM public.agent_sessions s '
                f'WHERE s.id = {table}.session_id AND s.user_id::text = {owner})')
    return f'user_id::text = {owner}'


def build_sql(owner_id, database, legacy_roles=None):
    """No passwords here. legacy_roles comes only from catalog preflight."""
    if not owner_id or '\x00' in owner_id:
        raise ValueError('An existing owner user ID is required.')
    role = ident(ROLE)
    statements = [
        f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{ROLE}') THEN CREATE ROLE {role}; END IF; END $$;",
        f'ALTER ROLE {role} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 20;',
        f"ALTER ROLE {role} SET search_path = public, pg_catalog;",
        f"ALTER ROLE {role} SET statement_timeout = '30s';",
        f"ALTER ROLE {role} SET idle_in_transaction_session_timeout = '30s';",
        f'REVOKE ALL ON DATABASE {ident(database)} FROM {role};',
        f'GRANT CONNECT ON DATABASE {ident(database)} TO {role};',
        f'REVOKE ALL ON SCHEMA public FROM {role};',
        f'GRANT USAGE ON SCHEMA public TO {role};',
        f'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role};',
        f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {role};',
        f'REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM {role};',
    ]
    for table in ALL_TABLES:
        qualified = f'public.{ident(table)}'
        predicate = owner_predicate(table, owner_id)
        statements.append(f'ALTER TABLE {qualified} ENABLE ROW LEVEL SECURITY;')
        # Only tables that previously had no RLS receive preservation policies.
        # Each role already had privileges; these policies add no table grants.
        for index, old_role in enumerate((legacy_roles or {}).get(table, ())):
            policy = ident(f'public_existing_access_{index}')
            statements.extend([
                f'DROP POLICY IF EXISTS {policy} ON {qualified};',
                f'CREATE POLICY {policy} ON {qualified} AS PERMISSIVE FOR ALL TO {ident(old_role)} USING (true) WITH CHECK (true);',
            ])
        for kind in ('PERMISSIVE', 'RESTRICTIVE'):
            policy = ident('public_owner_' + kind.lower())
            statements.extend([
                f'DROP POLICY IF EXISTS {policy} ON {qualified};',
                f'CREATE POLICY {policy} ON {qualified} AS {kind} FOR ALL TO {role} USING ({predicate}) WITH CHECK ({predicate});',
            ])
        privileges = WRITE_GRANTS.get(table, 'SELECT')
        statements.append(f'GRANT {privileges} ON {qualified} TO {role};')
    statements.extend(f'GRANT USAGE ON SEQUENCE public.{ident(s)} TO {role};' for s in SEQUENCES)
    return '\n'.join(statements)


def preflight(conn, owner_id):
    """Read-only catalog checks; refuses role reuse with elevated capabilities."""
    if not conn.execute('SELECT 1 FROM public.users WHERE id = %s', (owner_id,)).fetchone():
        raise ValueError('Owner must already exist in users.')
    role = conn.execute('SELECT oid FROM pg_roles WHERE rolname = %s', (ROLE,)).fetchone()
    if role:
        oid = role[0]
        if conn.execute('SELECT 1 FROM pg_auth_members WHERE member = %s OR roleid = %s', (oid, oid)).fetchone():
            raise ValueError('Dedicated role must have no role memberships or members.')
        if conn.execute('SELECT 1 FROM pg_class WHERE relowner = %s UNION ALL SELECT 1 FROM pg_namespace WHERE nspowner = %s UNION ALL SELECT 1 FROM pg_database WHERE datdba = %s UNION ALL SELECT 1 FROM pg_proc WHERE proowner = %s', (oid, oid, oid, oid)).fetchone():
            raise ValueError('Dedicated role must not own database objects.')
    legacy = {}
    for table in ALL_TABLES:
        row = conn.execute('SELECT oid, relrowsecurity FROM pg_class WHERE oid = %s::regclass', ('public.' + table,)).fetchone()
        if not row:
            raise ValueError('Required table is absent: ' + table)
        required = 'id' if table == 'users' else 'conversation_id' if table == 'conversation_messages' else 'session_id' if table in ('agent_events', 'agent_artifacts') else 'user_id'
        if not conn.execute('SELECT 1 FROM pg_attribute WHERE attrelid = %s AND attname = %s AND NOT attisdropped', (row[0], required)).fetchone():
            raise ValueError('Required ownership column is absent: ' + table)
        if not row[1]:
            roles = conn.execute("SELECT rolname FROM pg_roles WHERE NOT rolsuper AND rolname <> %s AND has_table_privilege(oid, %s, 'SELECT, INSERT, UPDATE, DELETE') ORDER BY rolname", (ROLE, row[0])).fetchall()
            legacy[table] = [r[0] for r in roles]
    return legacy


def verify_effective_access(conn):
    """Fail transaction if PUBLIC/inherited ACLs defeat the explicit allowlist."""
    if conn.execute("SELECT 1 FROM pg_namespace WHERE nspname NOT LIKE 'pg_%%' AND has_schema_privilege(%s, oid, 'CREATE')", (ROLE,)).fetchone():
        raise ValueError('PUBLIC schema CREATE must be removed by the administrator first.')
    if conn.execute("SELECT has_database_privilege(%s, current_database(), 'CREATE')", (ROLE,)).fetchone()[0]:
        raise ValueError('Dedicated role has inherited database CREATE access.')
    rows = conn.execute("SELECT n.nspname, c.relname, p.priv FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace CROSS JOIN (VALUES ('SELECT'), ('INSERT'), ('UPDATE'), ('DELETE'), ('TRUNCATE'), ('REFERENCES'), ('TRIGGER')) p(priv) WHERE n.nspname NOT LIKE 'pg_%%' AND n.nspname <> 'information_schema' AND c.relkind IN ('r','p','v','m','f') AND has_table_privilege(%s, c.oid, p.priv)", (ROLE,)).fetchall()
    for schema, table, privilege in rows:
        if (schema, table) in EXTENSION_METADATA and privilege == 'SELECT':
            continue
        allowed = WRITE_GRANTS.get(table, 'SELECT' if table in READ_TABLES else '') if schema == 'public' else ''
        if privilege not in allowed.split(', '):
            raise ValueError('Unexpected effective table grant: ' + table + ' ' + privilege)
    if conn.execute("SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE p.prosecdef AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND has_function_privilege(%s, p.oid, 'EXECUTE')", (ROLE,)).fetchone():
        raise ValueError('Accessible SECURITY DEFINER functions require explicit administrator review.')


def password_file(path):
    """Never follow links, overwrite existing credentials, or print a password."""
    flags = os.O_RDWR | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError('Password file must be owned by the current user and mode 0600.')
            value = file.read(4096).strip()
            if len(value) < 43 or len(value) > 512:
                raise ValueError('Existing password file must contain a strong generated password.')
            return value
    with os.fdopen(fd, 'w') as file:
        value = secrets.token_urlsafe(48)
        file.write(value + '\n')
        file.flush()
        os.fsync(file.fileno())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-id', required=True)
    parser.add_argument('--password-file', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply and args.password_file is None:
        parser.error('--apply requires --password-file')
    import psycopg
    from psycopg import sql
    dsn = os.environ.get('GMAIL_PROVISION_DSN')
    if not dsn:
        parser.error('Set GMAIL_PROVISION_DSN to an administrator connection string.')
    try:
        with psycopg.connect(dsn) as conn:
            legacy = preflight(conn, args.owner_id)
            database = conn.execute('SELECT current_database()').fetchone()[0]
            script = build_sql(args.owner_id, database, legacy)
            if not args.apply:
                print('-- Preview only; no changes made. Password intentionally omitted.\n' + script)
                conn.rollback()
                return
            conn.execute(script)
            verify_effective_access(conn)
            password = password_file(args.password_file)
            conn.execute(sql.SQL('ALTER ROLE {} PASSWORD {}').format(sql.Identifier(ROLE), sql.Literal(password)))
        print('Provisioned owner-scoped public database role. Password retained in the specified file.')
    except Exception as exc:
        # SQL/provider exceptions can contain credentials; report type only.
        if isinstance(exc, ValueError):
            parser.exit(1, str(exc) + '\n')
        parser.exit(1, 'Provisioning failed (' + type(exc).__name__ + '); transaction rolled back.\n')


if __name__ == '__main__':
    main()
