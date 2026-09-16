"""Native ranking must depend only on the authenticated owner's corpus."""
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

from gmail_search.gateway.partitions import partition_name, provision_owner_partitions
from test_gateway_partition_provision import partition_database as partition_database_fixture

partition_database = partition_database_fixture


@pytest.fixture
def ranked_database(partition_database):
    role = 'gms_rank_' + secrets.token_hex(12)
    password = secrets.token_urlsafe(40)
    with psycopg.connect(partition_database, autocommit=True) as admin:
        provision_owner_partitions(admin, 'alice')
        provision_owner_partitions(admin, 'bob')
        admin.execute(sql.SQL('CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}').format(sql.Identifier(role), sql.Literal(password)))
        try:
            admin.execute(sql.SQL('GRANT USAGE ON SCHEMA public,paradedb,pdb TO {}').format(sql.Identifier(role)))
            for table in ('messages','attachments','propositions'):
                admin.execute(sql.SQL('GRANT SELECT ON public.{} TO {}').format(sql.Identifier(table), sql.Identifier(role)))
                admin.execute(sql.SQL('CREATE POLICY legacy ON public.{} USING(true)').format(sql.Identifier(table)))
                admin.execute(sql.SQL("CREATE POLICY fixed_owner ON public.{} AS RESTRICTIVE TO {} USING(user_id='alice')").format(sql.Identifier(table), sql.Identifier(role)))
            for owner in ('alice','bob'):
                admin.execute("INSERT INTO public.messages(user_id,id,search_id,body_text) VALUES (%s,'same1',1,'alpha'),(%s,'same2',2,'beta filler')", (owner,owner))
                admin.execute("INSERT INTO public.attachments(user_id,message_id,id,filename,extracted_text) VALUES (%s,'same1',1,'first','alpha'),(%s,'same2',2,'second','beta filler')", (owner,owner))
                admin.execute("INSERT INTO public.propositions(user_id,message_id,id,text) VALUES (%s,'same1',1,'alpha'),(%s,'same2',2,'beta filler')", (owner,owner))
            yield admin, make_conninfo(partition_database, user=role, password=password), role
        finally:
            admin.execute(sql.SQL('DROP OWNED BY {}').format(sql.Identifier(role)))
            admin.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))


def relations(plan):
    found = []
    if isinstance(plan, dict):
        if 'Relation Name' in plan:
            found.append(plan['Relation Name'])
        for value in plan.values():
            found.extend(relations(value))
    elif isinstance(plan, list):
        for value in plan:
            found.extend(relations(value))
    return found


@pytest.mark.parametrize('table,key,field', [('messages','search_id','body_text'), ('attachments','id','extracted_text'), ('propositions','id','text')])
def test_rank_score_count_and_pruning_ignore_foreign_insert_update_delete(ranked_database, table, key, field):
    admin, reader_dsn, role = ranked_database
    query = sql.SQL('SELECT {},paradedb.score({}) FROM public.{} WHERE {} OPERATOR(pg_catalog.@@@) %s ORDER BY paradedb.score({}) DESC,{} LIMIT %s').format(
        *map(sql.Identifier, (key,key,table,key,key,key)))
    count = sql.SQL('SELECT count(*) FROM public.{} WHERE {} OPERATOR(pg_catalog.@@@) %s').format(sql.Identifier(table), sql.Identifier(key))
    term = f'{field}:alpha OR {field}:beta'
    with psycopg.connect(reader_dsn, autocommit=True, prepare_threshold=0) as reader:
        assert reader.execute('SELECT session_user,current_user').fetchone() == (role,role)
        reader.execute("SELECT set_config('app.user_id','bob',false)")
        baseline = reader.execute(query, (term,10)).fetchall()
        assert {row[0] for row in baseline} == {1,2}
        for mode in ('force_custom_plan','force_generic_plan'):
            reader.execute(sql.SQL('SET plan_cache_mode={}').format(sql.Literal(mode)))
            # Populate only Bob's rows, retaining equal numeric IDs across owners.
            if table == 'messages':
                admin.execute("INSERT INTO public.messages(user_id,id,search_id,body_text) SELECT 'bob','foreign'||i,100+i,'alpha foreignonly' FROM generate_series(1,200) i")
            elif table == 'attachments':
                admin.execute("INSERT INTO public.attachments(user_id,message_id,id,filename,extracted_text) SELECT 'bob','same1',100+i,'foreign'||i,'alpha foreignonly' FROM generate_series(1,200) i")
            else:
                admin.execute("INSERT INTO public.propositions(user_id,message_id,id,text) SELECT 'bob','same1',100+i,'alpha foreignonly' FROM generate_series(1,200) i")
            for mutation in ('insert','update','delete'):
                if mutation == 'update':
                    admin.execute(sql.SQL("UPDATE public.{} SET {}='beta foreignonly' WHERE user_id='bob' AND {}>100").format(*map(sql.Identifier,(table,field,key))))
                elif mutation == 'delete':
                    admin.execute(sql.SQL("DELETE FROM public.{} WHERE user_id='bob' AND {}>100").format(sql.Identifier(table),sql.Identifier(key)))
                assert reader.execute(query,(term,10)).fetchall() == baseline
                assert reader.execute(count,(term,)).fetchone() == (2,)
                assert reader.execute(query,(f'{field}:foreignonly',10)).fetchall() == []
                assert len(reader.execute(query,(term,1)).fetchall()) == 1
            plan = reader.execute(sql.SQL('EXPLAIN (ANALYZE,VERBOSE,FORMAT JSON) ') + query,(term,10)).fetchone()[0]
            assert set(relations(plan)) == {partition_name(table,'alice')}
        for owner in ('alice','bob'):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                reader.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier('gms_mail_partitions',partition_name(table,owner))))
