"""Administrator-only provisioning of immutable analytical reader identities.

Never import this into the guest or provide its administrator connection to
the public query handler. Changes are transactional and scoped to one reader.
"""
from __future__ import annotations

from psycopg import sql

from .database import reader_role
from .schema import ANALYTICAL_SCHEMA


def _binding(owner_id):
    return 'gmail-search analytical reader v1 owner=' + owner_id


def _preflight(conn, owner_id, role):
    if not conn.execute('SELECT 1 FROM public.users WHERE id=%s', (owner_id,)).fetchone():
        raise ValueError('Owner must already exist')
    current = conn.execute(
        "SELECT oid, rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls, shobj_description(oid, 'pg_authid') FROM pg_roles WHERE rolname=%s",
        (role,),
    ).fetchone()
    if current:
        if current[-1] != _binding(owner_id):
            raise ValueError('Existing reader owner binding does not match')
        if any(current[1:6]):
            raise ValueError('Existing reader has privileged attributes')
        oid = current[0]
        if conn.execute('SELECT 1 FROM pg_auth_members WHERE member=%s OR roleid=%s', (oid, oid)).fetchone():
            raise ValueError('Reader cannot have role memberships or members')
        if conn.execute("SELECT 1 FROM pg_shdepend WHERE refclassid='pg_authid'::regclass AND refobjid=%s AND deptype='o'", (oid,)).fetchone():
            raise ValueError('Reader cannot own database objects')
    for table, columns in ANALYTICAL_SCHEMA.items():
        relation = conn.execute("SELECT oid, relkind, relrowsecurity FROM pg_class WHERE oid=to_regclass(%s)", ('public.' + table,)).fetchone()
        if not relation or relation[1] not in ('r', 'p') or not relation[2]:
            raise ValueError('Analytical tables must already exist with RLS enabled: ' + table)
        actual = dict(conn.execute("SELECT a.attname, t.typname FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid JOIN pg_namespace n ON n.oid=t.typnamespace WHERE a.attrelid=%s AND a.attnum>0 AND NOT a.attisdropped AND n.nspname='pg_catalog'", (relation[0],)).fetchall())
        if 'user_id' not in columns or any(actual.get(column) != kind for column, kind in columns.items()):
            raise ValueError('Analytical column/type mismatch: ' + table)
    return bool(current)


def verify_reader_access(conn, role):
    if conn.execute("SELECT 1 FROM pg_class WHERE CASE WHEN relkind='S' THEN has_sequence_privilege(%s,oid,'SELECT,USAGE,UPDATE') ELSE false END", (role,)).fetchone():
        raise ValueError('Unexpected effective sequence grant')
    if conn.execute("SELECT has_database_privilege(%s,current_database(),'CREATE') OR has_database_privilege(%s,current_database(),'TEMP')", (role, role)).fetchone()[0]:
        raise ValueError('Reader inherits database CREATE/TEMP; administrator must review PUBLIC grants')
    if conn.execute("SELECT 1 FROM pg_namespace WHERE nspname NOT LIKE 'pg_%%' AND has_schema_privilege(%s,oid,'CREATE')", (role,)).fetchone():
        raise ValueError('Reader inherits schema CREATE grant')
    if conn.execute("SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE p.prosecdef AND n.nspname NOT IN ('pg_catalog','information_schema') AND has_function_privilege(%s,p.oid,'EXECUTE')", (role,)).fetchone():
        raise ValueError('Reader can execute a SECURITY DEFINER function')
    for schema, table, column, privilege in conn.execute("""
        SELECT n.nspname,c.relname,a.attname,p.priv
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
        CROSS JOIN (VALUES ('SELECT'),('INSERT'),('UPDATE'),('REFERENCES')) p(priv)
        WHERE c.relkind IN ('r','p','v','m','f')
        AND n.nspname NOT LIKE 'pg_%%' AND n.nspname <> 'information_schema'
        AND has_column_privilege(%s,c.oid,a.attnum,p.priv)
    """, (role,)):
        if not (schema == 'public' and privilege == 'SELECT' and column in ANALYTICAL_SCHEMA.get(table, {})):
            raise ValueError('Unexpected effective column grant: ' + schema + '.' + table + '.' + column)
    if conn.execute("""SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname NOT LIKE 'pg_%%' AND n.nspname <> 'information_schema'
        AND c.relkind IN ('r','p','v','m','f')
        AND has_table_privilege(%s,c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')""", (role,)).fetchone():
        raise ValueError('Unexpected effective write grant')


def provision_reader(conn, owner_id: str, password: str) -> str:
    """Create or rotate one reader; refuses unrelated ACL repairs and role reuse."""
    role = reader_role(owner_id)
    if not isinstance(password, str) or not 43 <= len(password) <= 512 or '\x00' in password:
        raise ValueError('A strong generated reader password is required')
    with conn.transaction():
        # Serialize competing provisioning for this identity, not all users.
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))', (role,))
        exists = _preflight(conn, owner_id, role)
        identifier = sql.Identifier(role)
        if not exists:
            conn.execute(sql.SQL('CREATE ROLE {} NOLOGIN').format(identifier))
            conn.execute(sql.SQL('COMMENT ON ROLE {} IS {}').format(identifier, sql.Literal(_binding(owner_id))))
        conn.execute(sql.SQL('ALTER ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 4').format(identifier))
        for setting, value in (
            ('search_path', 'pg_catalog'), ('statement_timeout', '10s'),
            ('lock_timeout', '1s'), ('idle_in_transaction_session_timeout', '10s'),
            ('work_mem', '4MB'), ('temp_file_limit', '64MB'),
            ('default_transaction_read_only', 'on'), ('max_parallel_workers_per_gather', '0'),
        ):
            conn.execute(sql.SQL('ALTER ROLE {} SET {} = {}').format(identifier, sql.Identifier(setting), sql.Literal(value)))
        conn.execute(sql.SQL('GRANT CONNECT ON DATABASE {} TO {}').format(sql.Identifier(conn.info.dbname), identifier))
        conn.execute(sql.SQL('GRANT USAGE ON SCHEMA public TO {}').format(identifier))
        for table, columns in ANALYTICAL_SCHEMA.items():
            qualified = sql.Identifier('public', table)
            for kind in ('PERMISSIVE', 'RESTRICTIVE'):
                policy = sql.Identifier(role + ('_allow' if kind == 'PERMISSIVE' else '_bound'))
                conn.execute(sql.SQL('DROP POLICY IF EXISTS {} ON {}').format(policy, qualified))
                conn.execute(sql.SQL('CREATE POLICY {} ON {} AS {} FOR SELECT TO {} USING (user_id = {})').format(policy, qualified, sql.SQL(kind), identifier, sql.Literal(owner_id)))
            conn.execute(sql.SQL('GRANT SELECT ({}) ON {} TO {}').format(sql.SQL(', ').join(map(sql.Identifier, columns)), qualified, identifier))
        verify_reader_access(conn, role)
        conn.execute(sql.SQL('ALTER ROLE {} PASSWORD {}').format(identifier, sql.Literal(password)))
    return role
