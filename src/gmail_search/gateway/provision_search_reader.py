"""Administrator provisioning and shared catalog audit for internal search logins.

No global ACL repair: unsafe PUBLIC privileges are a refusal, not an invitation
for this function to revoke other applications' grants.
"""
from psycopg import sql

from .partitions import PARTITION_SCHEMA, partition_binding, partition_name, verify_owner_partitions, _operation as partition_operation
from .search_reader import ROLE_SETTINGS, reader_binding, search_role, search_columns, search_functions
from .partition_profiles import NUMERIC_OWNER_PARTITIONS_V1 as NUMERIC, TEXT_OWNER_PARTITIONS_V1 as TEXT, require_profile


def _checks(owner_id, *, profile=NUMERIC):
    """Catalog-only generator shared by synchronous admin and async runtime.

    Each yielded (SQL, parameters) must receive a complete catalog result list.
    It reads no users/mail rows and never modifies the runtime connection.
    """
    require_profile(profile)
    columns_profile=search_columns(profile)
    role=search_role(owner_id, profile=profile)
    rows=yield ('''SELECT oid,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls,rolinherit,rolcanlogin,
        rolconnlimit,shobj_description(oid,'pg_authid') FROM pg_roles WHERE rolname=%s''',(role,))
    if len(rows)!=1 or rows[0][1:]!=(False,False,False,False,False,False,True,4,reader_binding(owner_id, profile=profile)):
        raise ValueError('Unqualified internal reader identity')
    role_oid=rows[0][0]
    rows=yield ('SELECT rolconfig FROM pg_roles WHERE oid=%s',(role_oid,))
    # Includes force_custom_plan. Old/mutated roles are refused before
    # rotation; provisioning never silently repairs their planner policy.
    if len(rows)!=1 or dict(item.split('=',1) for item in (rows[0][0] or []))!=dict(ROLE_SETTINGS):
        raise ValueError('Unqualified search role settings')
    rows=yield ("SELECT 1 FROM pg_auth_members WHERE member=%s OR roleid=%s UNION ALL SELECT 1 FROM pg_shdepend WHERE refclassid='pg_authid'::regclass AND refobjid=%s AND deptype='o' LIMIT 1",(role_oid,role_oid,role_oid))
    if rows:raise ValueError('Reader owns objects or has membership')
    rows=yield ("SELECT has_database_privilege(%s,current_database(),'CREATE,TEMP')",(role,))
    if rows!=[(False,)]:raise ValueError('Unexpected database privilege')
    rows=yield ("SELECT 1 FROM pg_class WHERE CASE WHEN relkind='S' THEN has_sequence_privilege(%s,oid,'SELECT,USAGE,UPDATE') ELSE false END LIMIT 1",(role,))
    if rows:raise ValueError('Unexpected sequence privilege')
    for statement,params in (
        ("SELECT 1 WHERE has_database_privilege(%s,current_database(),'CONNECT WITH GRANT OPTION')",(role,)),
        ("SELECT 1 FROM pg_namespace WHERE has_schema_privilege(%s,oid,'USAGE WITH GRANT OPTION,CREATE WITH GRANT OPTION') LIMIT 1",(role,)),
        ("SELECT 1 FROM pg_proc WHERE has_function_privilege(%s,oid,'EXECUTE WITH GRANT OPTION') LIMIT 1",(role,)),
        ('''SELECT 1 FROM pg_class c JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
            WHERE c.relkind IN ('r','p','v','m','f') AND has_column_privilege(%s,c.oid,a.attnum,
                'SELECT WITH GRANT OPTION,INSERT WITH GRANT OPTION,UPDATE WITH GRANT OPTION,REFERENCES WITH GRANT OPTION') LIMIT 1''',(role,)),
    ):
        rows=yield (statement,params)
        if rows:raise ValueError('Unexpected effective grant option')
    rows=yield ('''SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relkind IN ('r','p','v','m','f') AND n.nspname NOT LIKE 'pg_%%' AND n.nspname<>'information_schema'
        AND NOT(n.nspname='public' AND c.relname=ANY(%s)) AND has_table_privilege(%s,c.oid,'SELECT') LIMIT 1''',(list(columns_profile),role))
    if rows:raise ValueError('Unexpected relation SELECT privilege')
    rows=yield ("SELECT 1 FROM pg_namespace WHERE nspname NOT LIKE 'pg_%%' AND (has_schema_privilege(%s,oid,'CREATE') OR (nspname NOT IN ('public','paradedb','pdb','information_schema') AND has_schema_privilege(%s,oid,'USAGE'))) LIMIT 1",(role,role))
    if rows:raise ValueError('Unexpected schema privilege')
    rows=yield ('''SELECT n.nspname,c.relname,a.attname,p.priv FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
        CROSS JOIN (VALUES ('SELECT'),('INSERT'),('UPDATE'),('REFERENCES')) p(priv)
        WHERE c.relkind IN ('r','p','v','m','f') AND n.nspname NOT LIKE 'pg_%%' AND n.nspname<>'information_schema'
        AND has_column_privilege(%s,c.oid,a.attnum,p.priv)''',(role,))
    expected={('public',table,column,'SELECT') for table,columns in columns_profile.items() for column in columns}
    if set(rows)!=expected:raise ValueError('Unexpected effective search column privilege')
    rows=yield ('''SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname NOT LIKE 'pg_%%' AND n.nspname<>'information_schema' AND c.relkind IN ('r','p','v','m','f')
        AND has_table_privilege(%s,c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') LIMIT 1''',(role,))
    if rows:raise ValueError('Unexpected write privilege')
    rows=yield ("SELECT extversion,extowner FROM pg_extension WHERE extname='pg_search'",())
    if len(rows)!=1 or rows[0][0]!='0.23.0':raise ValueError('Unqualified search extension')
    administrator=rows[0][1]
    function_oids=[]
    for signature,expected in search_functions(profile).items():
        rows=yield ('''SELECT p.oid,p.prorettype::regtype::text,l.lanname,p.prosrc,p.probin,p.provolatile,
            p.prosecdef,p.proleakproof,p.proparallel,p.prokind,p.proowner,
            EXISTS(SELECT 1 FROM pg_depend d JOIN pg_extension e ON e.oid=d.refobjid
                WHERE d.classid='pg_proc'::regclass AND d.objid=p.oid AND d.refclassid='pg_extension'::regclass AND d.deptype='e' AND e.extname='pg_search'),
            has_function_privilege(%s,p.oid,'EXECUTE') FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
            WHERE p.oid=to_regprocedure(%s)''',(role,signature))
        if len(rows)!=1 or rows[0][1:]!=(*expected,administrator,True,True):
            raise ValueError('Unqualified search function identity/grant')
        function_oids.append(rows[0][0])
    if profile is TEXT:
        rows=yield ("SELECT proisstrict FROM pg_proc WHERE oid=to_regprocedure('paradedb.terms_with_operator(paradedb.fieldname,text,anyelement,boolean)')",())
        if rows!=[(True,)]:raise ValueError('Unqualified TEXT planner helper strictness')
    rows=yield ('''SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
        WHERE has_function_privilege(%s,p.oid,'EXECUTE') AND NOT p.oid=ANY(%s::oid[]) AND
        (n.nspname NOT IN ('pg_catalog','information_schema') OR EXISTS(SELECT 1 FROM pg_depend d JOIN pg_extension e ON e.oid=d.refobjid
            WHERE d.classid='pg_proc'::regclass AND d.objid=p.oid AND d.refclassid='pg_extension'::regclass AND e.extname='pg_search')) LIMIT 1''',(role,function_oids))
    if rows:raise ValueError('Unexpected effective function privilege')
    rows=yield ('''SELECT o.oprcode=to_regprocedure('paradedb.search_with_parse(anyelement,text)'),o.oprowner=%s,
        EXISTS(SELECT 1 FROM pg_depend d JOIN pg_extension e ON e.oid=d.refobjid WHERE d.classid='pg_operator'::regclass
            AND d.objid=o.oid AND d.refclassid='pg_extension'::regclass AND d.deptype='e' AND e.extname='pg_search')
        FROM pg_operator o WHERE o.oprnamespace='pg_catalog'::regnamespace AND o.oprname='@@@'
            AND o.oprleft='anyelement'::regtype AND o.oprright='text'::regtype''',(administrator,))
    if rows!=[(True,True,True)]:raise ValueError('Unqualified search operator')
    expression="(user_id = '"+owner_id.replace("'","''")+"'::text)"
    for table,columns in columns_profile.items():
        rows=yield ('''SELECT c.oid,c.relkind,c.relowner,c.relrowsecurity,c.relforcerowsecurity,c.relpersistence
            FROM pg_class c WHERE c.oid=to_regclass(%s)''',('public.'+table,))
        if len(rows)!=1 or rows[0][1:]!=(('p' if table in ('messages','attachments','propositions') else 'r'),administrator,True,True,'p'):
            raise ValueError('Unqualified search relation')
        oid=rows[0][0]
        rows=yield ('''SELECT a.attname,n.nspname,t.typname,a.attcollation=0 OR
            (a.attcollation='pg_catalog."default"'::regcollation AND col.collisdeterministic)
            FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid JOIN pg_namespace n ON n.oid=t.typnamespace
            LEFT JOIN pg_collation col ON col.oid=a.attcollation WHERE a.attrelid=%s AND a.attname=ANY(%s) AND NOT a.attisdropped''',(oid,list(columns)))
        if set(rows)!={(name,'pg_catalog',kind,True) for name,kind in columns.items()}:
            raise ValueError('Unqualified search column profile')
        rows=yield ('''SELECT polname,polpermissive,polcmd,polroles,pg_get_expr(polqual,polrelid),polwithcheck IS NULL
            FROM pg_policy WHERE polrelid=%s AND polname=ANY(%s)''',(oid,[role+'_allow',role+'_bound']))
        if sorted(rows)!=sorted([(role+'_allow',True,'r',[role_oid],expression,True),(role+'_bound',False,'r',[role_oid],expression,True)]):
            raise ValueError('Unqualified immutable owner policies')
    # TEXT must not quietly accept a numeric allocator left on the parent.
    if profile is TEXT:
        rows=yield ("SELECT 1 FROM pg_attribute WHERE attrelid='public.messages'::regclass AND attname='search_id' AND NOT attisdropped",())
        if rows:raise ValueError('Unqualified TEXT message schema')
    for table,(index,fields,key) in profile.indexes.items():
        child=partition_name(table,owner_id)
        rows=yield ('''SELECT c.oid,c.relkind,c.relowner,c.relrowsecurity,c.relforcerowsecurity,c.relpersistence,
            pg_get_expr(c.relpartbound,c.oid),obj_description(c.oid,'pg_class'),p.partstrat,p.partnatts,
            p.partexprs IS NULL,pg_get_partkeydef(p.partrelid),
            p.partcollation[0]='pg_catalog."default"'::regcollation::oid,
            (SELECT opcname='text_ops' AND opcdefault FROM pg_opclass WHERE oid=p.partclass[0])
            FROM pg_class c JOIN pg_namespace ns ON ns.oid=c.relnamespace JOIN pg_inherits h ON h.inhrelid=c.oid JOIN pg_partitioned_table p ON p.partrelid=h.inhparent
            WHERE ns.nspname=%s AND c.relname=%s AND h.inhparent=to_regclass(%s)''',(PARTITION_SCHEMA,child,'public.'+table))
        bound="FOR VALUES IN ('"+owner_id.replace("'","''")+"')"
        if len(rows)!=1 or rows[0][1:]!=('r',administrator,True,True,'p',bound,partition_binding(table,owner_id),'l',1,True,'LIST (user_id)',True,True):
            raise ValueError('Unqualified owner search partition')
        child_oid=rows[0][0]
        rows=yield ('''SELECT c.relkind,c.relowner,i.indisvalid,i.indisready,i.indislive,i.indisunique,i.indexprs IS NULL,i.indpred IS NULL,
            i.indnkeyatts=i.indnatts,ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY k(num,n)
                JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.num ORDER BY k.n),c.reloptions,
            NOT EXISTS(SELECT 1 FROM unnest(i.indclass::oid[]) op(oid) JOIN pg_opclass o ON o.oid=op.oid WHERE NOT o.opcdefault),
            i.indoption::smallint[] FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am am ON am.oid=c.relam
            WHERE am.amname='bm25' AND (i.indexrelid=to_regclass(%s) OR (i.indrelid=%s::oid
                AND EXISTS(SELECT 1 FROM pg_inherits h WHERE h.inhrelid=i.indexrelid AND h.inhparent=to_regclass(%s)))) ORDER BY c.relkind''',('public.'+index,child_oid,'public.'+index))
        common=(administrator,True,True,True,False,True,True,True,list(fields),['key_field='+key],True,[0]*len(fields))
        if rows!=[('I',*common),('i',*common)]:raise ValueError('Unqualified native owner index')


def verify_search_reader_access(conn,owner_id, *, profile=NUMERIC):
    checks=_checks(owner_id, profile=profile)
    try:
        request=next(checks)
        while True:
            statement,params=request
            request=checks.send(conn.execute(statement,params).fetchall())
    except StopIteration:
        return


def provision_search_reader(conn,owner_id,password, *, profile=NUMERIC):
    require_profile(profile)
    columns_profile=search_columns(profile)
    role=search_role(owner_id, profile=profile)
    if not isinstance(password,str) or not 43<=len(password)<=512 or '\x00' in password:
        raise ValueError('Strong generated search password required')
    # Bound the whole administrator operation, not only partition verification.
    # The shared scope preserves stricter caller limits and restores settings.
    with partition_operation(conn):
        verify_owner_partitions(conn,owner_id,profile=profile)
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(role,))
        existing=conn.execute("SELECT shobj_description(oid,'pg_authid') FROM pg_roles WHERE rolname=%s",(role,)).fetchone()
        ident=sql.Identifier(role)
        if existing:
            if existing!=(reader_binding(owner_id, profile=profile),):raise ValueError('Existing search role binding mismatch')
            # Refuse drift before rotating credentials; never repair a broadened role.
            verify_search_reader_access(conn,owner_id,profile=profile)
        else:
            conn.execute(sql.SQL('CREATE ROLE {} NOLOGIN').format(ident))
            conn.execute(sql.SQL('COMMENT ON ROLE {} IS {}').format(ident,sql.Literal(reader_binding(owner_id, profile=profile))))
        conn.execute(sql.SQL('ALTER ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT CONNECTION LIMIT 4').format(ident))
        for setting,value in ROLE_SETTINGS:
            conn.execute(sql.SQL('ALTER ROLE {} SET {} = {}').format(ident,sql.Identifier(setting),sql.Literal(value)))
        conn.execute(sql.SQL('GRANT CONNECT ON DATABASE {} TO {}').format(sql.Identifier(conn.info.dbname),ident))
        conn.execute(sql.SQL('GRANT USAGE ON SCHEMA public,paradedb,pdb TO {}').format(ident))
        # Embeddings have no existing guest grant. Enforce RLS here only, through
        # reviewed administrator provisioning, before granting internal columns.
        conn.execute('ALTER TABLE public.embeddings ENABLE ROW LEVEL SECURITY')
        conn.execute('ALTER TABLE public.embeddings FORCE ROW LEVEL SECURITY')
        for table,columns in columns_profile.items():
            target=sql.Identifier('public',table)
            for suffix,kind in (('_allow','PERMISSIVE'),('_bound','RESTRICTIVE')):
                policy=sql.Identifier(role+suffix)
                conn.execute(sql.SQL('DROP POLICY IF EXISTS {} ON {}').format(policy,target))
                conn.execute(sql.SQL('CREATE POLICY {} ON {} AS {} FOR SELECT TO {} USING(user_id={})').format(policy,target,sql.SQL(kind),ident,sql.Literal(owner_id)))
            conn.execute(sql.SQL('GRANT SELECT({}) ON {} TO {}').format(sql.SQL(',').join(map(sql.Identifier,columns)),target,ident))
        for signature in search_functions(profile):
            conn.execute(sql.SQL('GRANT EXECUTE ON FUNCTION {} TO {}').format(sql.SQL(signature),ident))
        verify_search_reader_access(conn,owner_id,profile=profile)
        conn.execute(sql.SQL('ALTER ROLE {} PASSWORD {}').format(ident,sql.Literal(password)))
    return role
