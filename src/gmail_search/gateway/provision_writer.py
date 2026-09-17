"""Administrator-only fixed-owner conversation writer provisioning.

This profile intentionally excludes mailbox data, users, sync_state, artifacts,
legacy runtime mappings and model battles. Required RLS must already be enabled
by a reviewed administrator migration; this module never changes legacy roles
or repairs PUBLIC grants implicitly.
"""
import json
from contextlib import contextmanager

from pglast import parser
from psycopg import sql

from .browser_conversations import validate_receipt_schema
from .schema import EXTENSION_METADATA
from .writer import application_writer_role,writer_binding

# Explicit current columns and built-in types; no whole-table SELECT/INSERT/UPDATE.
_COLUMNS={
    'browser_answer_receipts': dict.fromkeys('run_id owner_id conversation_id content_hash'.split(),'text'),
    'conversations': dict.fromkeys('id title created_at updated_at user_id'.split(),'text'),
    'conversation_messages': {**dict.fromkeys('conversation_id role parts created_at'.split(),'text'),'id':'int8','seq':'int8'},
    'agent_sessions': {**dict.fromkeys('id conversation_id mode question final_answer status user_id'.split(),'text'),'plan':'jsonb','started_at':'timestamptz','finished_at':'timestamptz'},
    'agent_events': {'id':'int8','session_id':'text','seq':'int4','agent_name':'text','kind':'text','payload':'jsonb','created_at':'timestamptz'},
    'costs': {**dict.fromkeys('timestamp operation model message_id user_id'.split(),'text'),**dict.fromkeys('id input_tokens image_count output_tokens cached_input_tokens cache_write_tokens'.split(),'int8'),'estimated_cost_usd':'float8'},
}
_INSERT={
    'browser_answer_receipts': tuple(_COLUMNS['browser_answer_receipts']),
    'conversations': tuple(_COLUMNS['conversations']),
    'conversation_messages': ('conversation_id','seq','role','parts','created_at'),
    'agent_sessions': ('id','conversation_id','mode','question','status','user_id'),
    'agent_events': ('session_id','seq','agent_name','kind','payload'),
    'costs': tuple(column for column in _COLUMNS['costs'] if column!='id'),
}
_UPDATE={'conversations':('title','updated_at'),'agent_sessions':('plan','status','final_answer','finished_at')}
_DELETE={'conversations','conversation_messages'}
_SEQUENCES={table:table+'_id_seq' for table in ('conversation_messages','agent_events','costs')}
_BINDING_PREFIX='gmail-search application writer v1 owner='


def _predicate(table,owner_id):
    owner=sql.Literal(owner_id)
    if table=='browser_answer_receipts':
        return sql.SQL('owner_id={} AND EXISTS(SELECT 1 FROM public.conversations c WHERE c.id=browser_answer_receipts.conversation_id AND c.user_id={})').format(owner,owner)
    if table=='conversation_messages':
        return sql.SQL('EXISTS(SELECT 1 FROM public.conversations c WHERE c.id=conversation_messages.conversation_id AND c.user_id={})').format(owner)
    if table=='agent_events':
        return sql.SQL('EXISTS(SELECT 1 FROM public.agent_sessions s WHERE s.id=agent_events.session_id AND s.user_id={})').format(owner)
    if table=='agent_sessions':
        return sql.SQL('user_id={} AND (conversation_id IS NULL OR EXISTS(SELECT 1 FROM public.conversations c WHERE c.id=agent_sessions.conversation_id AND c.user_id={}))').format(owner,owner)
    return sql.SQL('user_id={}').format(owner)


def _preflight(conn,owner_id,role):
    if not conn.execute('SELECT 1 FROM public.users WHERE id=%s',(owner_id,)).fetchone():
        raise ValueError('An existing canonical user is required')
    current=conn.execute("SELECT oid,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls,shobj_description(oid,'pg_authid') FROM pg_roles WHERE rolname=%s",(role,)).fetchone()
    if current:
        if current[-1]!=writer_binding(owner_id) or any(current[1:6]):
            raise ValueError('Existing writer identity is not safe to reuse')
        if conn.execute('SELECT 1 FROM pg_auth_members WHERE member=%s OR roleid=%s',(current[0],current[0])).fetchone():
            raise ValueError('Writer cannot have role memberships or members')
        if conn.execute("SELECT 1 FROM pg_shdepend WHERE refclassid='pg_authid'::regclass AND refobjid=%s AND deptype='o'",(current[0],)).fetchone():
            raise ValueError('Writer cannot own database objects')
    for table,columns in _COLUMNS.items():
        relation=conn.execute("SELECT oid,relkind,relrowsecurity FROM pg_class WHERE oid=to_regclass(%s)",('public.'+table,)).fetchone()
        if not relation or relation[1]!='r' or not relation[2]:
            raise ValueError('Application tables require existing ordinary-table RLS: '+table)
        if conn.execute('SELECT 1 FROM pg_trigger WHERE tgrelid=%s AND NOT tgisinternal UNION ALL SELECT 1 FROM pg_rewrite WHERE ev_class=%s UNION ALL SELECT 1 FROM pg_inherits WHERE inhrelid=%s OR inhparent=%s',(relation[0],)*4).fetchone():
            raise ValueError('Application table has unqualified triggers, rules or inheritance')
        actual=dict(conn.execute("SELECT a.attname,t.typname FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid JOIN pg_namespace n ON n.oid=t.typnamespace WHERE a.attrelid=%s AND a.attnum>0 AND NOT a.attisdropped AND n.nspname='pg_catalog'",(relation[0],)).fetchall())
        if any(actual.get(column)!=kind for column,kind in columns.items()):
            raise ValueError('Application writer column/type mismatch: '+table)
        if table in _SEQUENCES:
            sequence=conn.execute('SELECT pg_get_serial_sequence(%s,%s)',('public.'+table,'id')).fetchone()[0]
            if sequence!='public.'+_SEQUENCES[table]:
                raise ValueError('Application sequence identity mismatch')
    validate_receipt_schema(conn)
    return current is not None


def _normalized_expression(value):
    """Parse one policy expression and normalize only known deparser noise."""
    def string_values(items):
        values=[]
        for item in items or []:
            value=item.get('String',{}).get('sval') if isinstance(item,dict) else None
            if value is None:return None
            values.append(value)
        return values

    def normalize(node):
        if isinstance(node,list):return [normalize(item) for item in node]
        if not isinstance(node,dict):return node
        cast=node.get('TypeCast') if set(node)=={'TypeCast'} else None
        if cast:
            type_name=cast.get('typeName',{})
            names=string_values(type_name.get('names'))
            constant=cast.get('arg',{}).get('A_Const',{})
            plain_text=(set(type_name)<= {'names','typemod','location'} and
                        type_name.get('typemod',-1)==-1)
            if plain_text and names in (['text'],['pg_catalog','text']) and 'sval' in constant:
                return normalize(cast['arg'])
        normalized={}
        for key,item in node.items():
            if key in ('location','stmt_location','stmt_len'):continue
            normalized[key]=normalize(item)
        return normalized

    if not isinstance(value,str):raise ValueError('Unqualified immutable owner policies')
    try:
        document=json.loads(parser.parse_sql_json('SELECT '+value))
        target=document['stmts'][0]['stmt']['SelectStmt']['targetList']
        if len(document['stmts'])!=1 or len(target)!=1:
            raise ValueError
        expression=target[0]['ResTarget']['val']
    except Exception:
        raise ValueError('Unqualified immutable owner policies') from None
    return normalize(expression)


def _writer_identity(conn,role,owner_id):
    row=conn.execute("SELECT oid,shobj_description(oid,'pg_authid') FROM pg_roles WHERE rolname=%s",(role,)).fetchone()
    if not row or not isinstance(row[1],str) or not row[1].startswith(_BINDING_PREFIX):
        raise ValueError('Unqualified application writer identity')
    bound_owner=row[1][len(_BINDING_PREFIX):]
    if (not bound_owner or writer_binding(bound_owner)!=row[1] or
            application_writer_role(bound_owner)!=role or
            (owner_id is not None and owner_id!=bound_owner)):
        raise ValueError('Unqualified application writer identity')
    return row[0],bound_owner


@contextmanager
def _catalog_search_path(conn):
    """Deparse with no application schema visible, then always restore state."""
    with conn.transaction(force_rollback=True):
        conn.execute("SELECT set_config('search_path','pg_catalog',true)")
        yield


def _verify_writer_policies(conn,role,role_oid,owner_id):
    policy_names=[role+'_allow',role+'_bound']
    with _catalog_search_path(conn):
        for table in _COLUMNS:
            relation=conn.execute("SELECT oid,relkind,relrowsecurity,relowner FROM pg_class WHERE oid=to_regclass(%s)",('public.'+table,)).fetchone()
            if not relation or relation[1]!='r' or not relation[2] or relation[3]==role_oid:
                raise ValueError('Unqualified application writer RLS: '+table)
            rows=conn.execute('''SELECT polname,polpermissive,polcmd,polroles,
                pg_get_expr(polqual,polrelid),pg_get_expr(polwithcheck,polrelid)
                FROM pg_policy WHERE polrelid=%s AND polname=ANY(%s)''',(relation[0],policy_names)).fetchall()
            if len(rows)!=2 or {row[0] for row in rows}!=set(policy_names):
                raise ValueError('Unqualified immutable owner policies: '+table)
            expected_tree=_normalized_expression(_predicate(table,owner_id).as_string(conn))
            for name,permissive,command,roles,using,check in rows:
                expected_permissive=name.endswith('_allow')
                if (permissive is not expected_permissive or command!='*' or
                        roles!=[role_oid] or _normalized_expression(using)!=expected_tree or
                        _normalized_expression(check)!=expected_tree):
                    raise ValueError('Unqualified immutable owner policies: '+table)


def verify_writer_access(conn,role,owner_id=None):
    role_oid,owner_id=_writer_identity(conn,role,owner_id)
    _verify_writer_policies(conn,role,role_oid,owner_id)
    if conn.execute("SELECT has_database_privilege(%s,current_database(),'CREATE') OR has_database_privilege(%s,current_database(),'TEMP')",(role,role)).fetchone()[0]:
        raise ValueError('Writer inherits database CREATE/TEMP; review PUBLIC grants')
    if conn.execute("SELECT 1 FROM pg_namespace WHERE nspname NOT LIKE 'pg_%%' AND has_schema_privilege(%s,oid,'CREATE')",(role,)).fetchone():
        raise ValueError('Writer inherits schema CREATE')
    if conn.execute("SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE p.prosecdef AND n.nspname NOT IN ('pg_catalog','information_schema') AND has_function_privilege(%s,p.oid,'EXECUTE')",(role,)).fetchone():
        raise ValueError('Writer can execute a SECURITY DEFINER function')
    for schema,table,column,privilege in conn.execute("""SELECT n.nspname,c.relname,a.attname,p.priv
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
        CROSS JOIN(VALUES('SELECT'),('INSERT'),('UPDATE'),('REFERENCES'))p(priv)
        WHERE c.relkind IN('r','p','v','m','f') AND n.nspname NOT LIKE 'pg_%%' AND n.nspname<>'information_schema'
        AND has_column_privilege(%s,c.oid,a.attnum,p.priv)""",(role,)):
        allowed=_COLUMNS if privilege=='SELECT' else _INSERT if privilege=='INSERT' else _UPDATE if privilege=='UPDATE' else {}
        # Extension catalogues are PUBLIC-readable in this image; see EXTENSION_METADATA.
        if privilege=='SELECT' and (schema,table) in EXTENSION_METADATA:
            continue
        if schema!='public' or column not in allowed.get(table,{}):
            raise ValueError('Unexpected effective writer column grant: '+schema+'.'+table+'.'+column)
    for schema,table,privilege in conn.execute("""SELECT n.nspname,c.relname,p.priv FROM pg_class c
        JOIN pg_namespace n ON n.oid=c.relnamespace
        CROSS JOIN(VALUES('SELECT'),('INSERT'),('UPDATE'),('DELETE'),('TRUNCATE'),('REFERENCES'),('TRIGGER'))p(priv)
        WHERE c.relkind IN('r','p','v','m','f') AND n.nspname NOT LIKE 'pg_%%' AND n.nspname<>'information_schema'
        AND has_table_privilege(%s,c.oid,p.priv)""",(role,)):
        if privilege=='SELECT' and (schema,table) in EXTENSION_METADATA:
            continue
        if not(schema=='public' and table in _DELETE and privilege=='DELETE'):
            # Name the object: this refusal used to say only that something was
            # wrong, which is no help against a database with extension baggage.
            raise ValueError('Unexpected whole-table writer grant: '+schema+'.'+table+' '+privilege)
    for schema,sequence,privilege in conn.execute("""SELECT n.nspname,c.relname,p.priv FROM pg_class c
        JOIN pg_namespace n ON n.oid=c.relnamespace CROSS JOIN(VALUES('USAGE'),('SELECT'),('UPDATE'))p(priv)
        WHERE CASE WHEN c.relkind='S' THEN has_sequence_privilege(%s,c.oid,p.priv) ELSE false END""",(role,)):
        if schema!='public' or sequence not in _SEQUENCES.values() or privilege!='USAGE':
            raise ValueError('Unexpected effective writer sequence grant')


def provision_application_writer(conn,owner_id,password):
    """Create/rotate one owner role transactionally; accepts no request SQL."""
    role=application_writer_role(owner_id)
    if not isinstance(password,str) or not 43<=len(password)<=512 or '\x00' in password:
        raise ValueError('A strong generated writer password is required')
    with conn.transaction():
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(role,))
        existing=_preflight(conn,owner_id,role)
        identifier=sql.Identifier(role)
        if not existing:
            conn.execute(sql.SQL('CREATE ROLE {} NOLOGIN').format(identifier))
            conn.execute(sql.SQL('COMMENT ON ROLE {} IS {}').format(identifier,sql.Literal(writer_binding(owner_id))))
        conn.execute(sql.SQL('ALTER ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 4').format(identifier))
        for setting,value in (('search_path','public'),('statement_timeout','10s'),('lock_timeout','1s'),('idle_in_transaction_session_timeout','10s'),('work_mem','4MB'),('temp_file_limit','64MB'),('max_parallel_workers_per_gather','0')):
            conn.execute(sql.SQL('ALTER ROLE {} SET {}={}').format(identifier,sql.Identifier(setting),sql.Literal(value)))
        conn.execute(sql.SQL('GRANT CONNECT ON DATABASE {} TO {}').format(sql.Identifier(conn.info.dbname),identifier))
        conn.execute(sql.SQL('GRANT USAGE ON SCHEMA public TO {}').format(identifier))
        for table,columns in _COLUMNS.items():
            qualified=sql.Identifier('public',table)
            predicate=_predicate(table,owner_id)
            for kind in ('PERMISSIVE','RESTRICTIVE'):
                policy=sql.Identifier(role+('_allow' if kind=='PERMISSIVE' else '_bound'))
                conn.execute(sql.SQL('DROP POLICY IF EXISTS {} ON {}').format(policy,qualified))
                conn.execute(sql.SQL('CREATE POLICY {} ON {} AS {} FOR ALL TO {} USING({}) WITH CHECK({})').format(policy,qualified,sql.SQL(kind),identifier,predicate,predicate))
            for privilege,names in (('SELECT',columns),('INSERT',_INSERT[table]),('UPDATE',_UPDATE.get(table,()))):
                if names:
                    conn.execute(sql.SQL('GRANT {}({}) ON {} TO {}').format(sql.SQL(privilege),sql.SQL(',').join(map(sql.Identifier,names)),qualified,identifier))
            if table in _DELETE:
                conn.execute(sql.SQL('GRANT DELETE ON {} TO {}').format(qualified,identifier))
        for sequence in _SEQUENCES.values():
            conn.execute(sql.SQL('GRANT USAGE ON SEQUENCE {} TO {}').format(sql.Identifier('public',sequence),identifier))
        verify_writer_access(conn,role,owner_id)
        conn.execute(sql.SQL('ALTER ROLE {} PASSWORD {}').format(identifier,sql.Literal(password)))
    return role
