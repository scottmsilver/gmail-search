"""Synthetic pg_search/RLS probe, restricted to the disposable loopback:55440.

Run with GMS_RANKED_PROBE_DSN pointing to the approved disposable postgres DB.
Creates and finally drops only its random database and roles. No app imports,
production connection, existing reader ACL changes, or mailbox data are used.
Prints synthetic results/plans and catalog evidence as JSON, never credentials.
"""
import json
import os
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo


RANKED = """SELECT search_id,user_id,id,paradedb.score(search_id) AS score
FROM public.messages
WHERE search_id OPERATOR(pg_catalog.@@@) %s
ORDER BY paradedb.score(search_id) DESC LIMIT %s"""
COUNT = "SELECT count(*) FROM public.messages WHERE search_id OPERATOR(pg_catalog.@@@) %s"
DOCUMENT_RANKED = """SELECT search_id,user_id,id,
pg_catalog.ts_rank_cd(
  pg_catalog.to_tsvector('pg_catalog.english'::regconfig,coalesce(subject,'') || ' ' || body_text),
  pg_catalog.to_tsquery('pg_catalog.english'::regconfig,%s),2) AS lexical_rank
FROM public.messages WHERE search_id OPERATOR(pg_catalog.@@@) %s
ORDER BY lexical_rank DESC,id ASC LIMIT %s"""


def observe(conn, statement, params=()):
    try:
        cursor = conn.execute(statement, params)
        return {"rows": cursor.fetchall()} if cursor.description else {"ok": True}
    except psycopg.Error as error:
        # Diagnostics refer only to this probe's invented schema and terms.
        return {"sqlstate": error.sqlstate, "error": error.diag.message_primary}


def probe(dsn):
    cfg = conninfo_to_dict(dsn)
    if (cfg.get("host") != "127.0.0.1" or cfg.get("port") != "55440"
            or cfg.get("dbname") != "postgres" or cfg.get("user") != "postgres"
            or any(key in cfg for key in ("hostaddr", "service", "options"))):
        raise ValueError("Requires the approved disposable 127.0.0.1:55440/postgres DSN")
    name = "gms_ranked_probe_" + secrets.token_hex(8)
    roles = {owner: name + "_" + owner for owner in ("alice", "bob")}
    passwords = {owner: secrets.token_urlsafe(40) for owner in roles}
    created_roles = []
    target = make_conninfo(dsn, dbname=name, connect_timeout=5)
    report = {"ranked_query": RANKED, "document_ranked_query": DOCUMENT_RANKED, "count_query": COUNT, "owners": list(roles)}
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
    try:
        with psycopg.connect(target, autocommit=True) as admin:
            admin.execute("SET statement_timeout='10s'")
            admin.execute("CREATE EXTENSION pg_search")
            report["version"] = admin.execute("SELECT version()").fetchone()[0]
            report["extension"] = admin.execute("SELECT extname,extversion FROM pg_extension WHERE extname='pg_search'").fetchone()
            assert report["extension"] == ("pg_search", "0.23.0")
            report["operator"] = admin.execute("""SELECT n.nspname,o.oprname,
                format_type(o.oprleft,NULL),format_type(o.oprright,NULL),pn.nspname,p.proname,
                pg_get_function_identity_arguments(p.oid),p.proleakproof,p.prosecdef,e.extname
                FROM pg_operator o JOIN pg_namespace n ON n.oid=o.oprnamespace
                JOIN pg_proc p ON p.oid=o.oprcode JOIN pg_namespace pn ON pn.oid=p.pronamespace
                JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid AND d.deptype='e'
                JOIN pg_extension e ON e.oid=d.refobjid
                WHERE o.oprname='@@@' AND o.oprleft='anyelement'::regtype AND o.oprright='text'::regtype""").fetchall()
            report["score"] = admin.execute("""SELECT n.nspname,p.proname,
                pg_get_function_identity_arguments(p.oid),p.proleakproof,p.prosecdef
                FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                WHERE n.nspname='paradedb' AND p.proname='score'""").fetchall()
            admin.execute("""CREATE TABLE public.messages (
                search_id bigint PRIMARY KEY,user_id text NOT NULL,id text NOT NULL,
                body_text text NOT NULL,subject text DEFAULT '',withheld text DEFAULT 'not granted',UNIQUE(user_id,id))""")
            admin.execute("ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY")
            admin.execute("ALTER TABLE public.messages FORCE ROW LEVEL SECURITY")
            admin.execute("CREATE POLICY legacy ON public.messages USING(true)")
            # Revoke only inside this disposable database. This deliberately
            # does not use or weaken the application's analytical provisioning.
            admin.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(name)))
            schemas = admin.execute("SELECT nspname FROM pg_namespace WHERE nspname IN ('public','paradedb','pdb')").fetchall()
            for (schema,) in schemas:
                ident = sql.Identifier(schema)
                admin.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(ident))
                admin.execute(sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM PUBLIC").format(ident))
                admin.execute(sql.SQL("REVOKE ALL ON ALL SEQUENCES IN SCHEMA {} FROM PUBLIC").format(ident))
            functions = admin.execute("""SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid)
                FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid AND d.deptype='e'
                JOIN pg_extension e ON e.oid=d.refobjid WHERE e.extname='pg_search'""").fetchall()
            for schema, function, arguments in functions:
                admin.execute(sql.SQL("REVOKE EXECUTE ON ROUTINE {}.{}({}) FROM PUBLIC").format(
                    sql.Identifier(schema), sql.Identifier(function), sql.SQL(arguments)))
            report["revoked_extension_function_count"] = len(functions)
            for owner, role in roles.items():
                ident = sql.Identifier(role)
                admin.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT NOBYPASSRLS").format(ident, sql.Literal(passwords[owner])))
                created_roles.append(role)
                admin.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(name), ident))
                admin.execute(sql.SQL("GRANT USAGE ON SCHEMA public,paradedb,pdb TO {}").format(ident))
                admin.execute(sql.SQL("GRANT SELECT(search_id,user_id,id,subject,body_text) ON public.messages TO {}").format(ident))
                admin.execute(sql.SQL("CREATE POLICY {} ON public.messages AS RESTRICTIVE TO {} USING(user_id={})").format(sql.Identifier(owner + "_guard"), ident, sql.Literal(owner)))
                admin.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only=on").format(ident))
                admin.execute(sql.SQL("ALTER ROLE {} SET statement_timeout='10s'").format(ident))
            # Deliberately low-scoring Alice rows; colliding Gmail IDs in Bob.
            rows = [(i + 1, "alice", "same" + str(i),
                     ("alpha" if i == 0 else "beta filler" if i == 1 else "needle " + "filler " * 100)) for i in range(12)]
            with admin.cursor() as cur:
                cur.executemany("INSERT INTO public.messages(search_id,user_id,id,body_text) VALUES(%s,%s,%s,%s)", rows)
            admin.execute("CREATE INDEX messages_bm25_idx ON public.messages USING bm25(search_id,body_text) WITH(key_field='search_id')")
            def connect(owner):
                return psycopg.connect(make_conninfo(target, user=roles[owner], password=passwords[owner]), autocommit=True)
            with connect("alice") as alice:
                report["direct_identity"] = alice.execute("SELECT session_user=current_user,current_user=%s", (roles["alice"],)).fetchone()
                admin.execute(sql.SQL("REVOKE USAGE ON SCHEMA pdb FROM {}").format(sql.Identifier(roles["alice"])))
                report["without_pdb_usage"] = observe(alice, "SELECT id FROM public.messages")
                admin.execute(sql.SQL("GRANT USAGE ON SCHEMA pdb TO {}").format(sql.Identifier(roles["alice"])))
                report["without_function_grants"] = observe(alice, RANKED, ("body_text:needle", 10))
                report["without_operator_grant"] = observe(alice, COUNT, ("body_text:needle",))
                assert report["without_function_grants"].get("sqlstate") == "42501"
                # Begin with operator/score grants; prepared-plan helper grants
                # are added explicitly below, never by granting a whole schema.
                for role in roles.values():
                    admin.execute(sql.SQL("GRANT EXECUTE ON FUNCTION paradedb.search_with_parse(anyelement,text) TO {}").format(sql.Identifier(role)))
                report["operator_only_count"] = observe(alice, COUNT, ("body_text:needle",))
                report["without_score_grant"] = observe(alice, RANKED, ("body_text:needle", 10))
                for role in roles.values():
                    admin.execute(sql.SQL("GRANT EXECUTE ON FUNCTION paradedb.score(anyelement) TO {}").format(sql.Identifier(role)))
                admin.execute(sql.SQL("REVOKE EXECUTE ON FUNCTION paradedb.search_with_parse(anyelement,text) FROM {}").format(sql.Identifier(roles["alice"])))
                report["score_only_ranked"] = observe(alice, RANKED, ("body_text:needle", 10))
                admin.execute(sql.SQL("GRANT EXECUTE ON FUNCTION paradedb.search_with_parse(anyelement,text) TO {}").format(sql.Identifier(roles["alice"])))
                for role in roles.values():
                    admin.execute(sql.SQL("GRANT EXECUTE ON FUNCTION paradedb.with_index(regclass,paradedb.searchqueryinput) TO {}").format(sql.Identifier(role)))
                    admin.execute(sql.SQL("GRANT EXECUTE ON FUNCTION paradedb.parse_with_field(paradedb.fieldname,text,boolean,boolean) TO {}").format(sql.Identifier(role)))
                report["before_foreign_corpus"] = observe(alice, RANKED, ("body_text:alpha OR body_text:beta", 10))
                local_params = ("alpha | beta", "body_text:alpha OR body_text:beta", 10)
                report["document_rank_before_foreign"] = observe(alice, DOCUMENT_RANKED, local_params)
                foreign = [(1000 + i, "bob", "same" + str(i), "needle alpha foreignonly " * 20) for i in range(200)]
                with admin.cursor() as cur:
                    cur.executemany("INSERT INTO public.messages(search_id,user_id,id,body_text) VALUES(%s,%s,%s,%s)", foreign)
                admin.execute("VACUUM ANALYZE public.messages")
                report["after_foreign_corpus"] = observe(alice, RANKED, ("body_text:alpha OR body_text:beta", 10))
                report["document_rank_after_foreign"] = observe(alice, DOCUMENT_RANKED, local_params)
                report["document_rank_plan"] = observe(alice, "EXPLAIN (ANALYZE,VERBOSE,FORMAT JSON) " + DOCUMENT_RANKED, local_params)
                report["document_rank_foreign"] = observe(alice, DOCUMENT_RANKED, ("foreignonly", "body_text:foreignonly", 10))
                report["large_tsvector"] = observe(alice, "SELECT length(to_tsvector('pg_catalog.simple',string_agg('token'||i,' '))::text) FROM generate_series(1,150000) AS i")
                report["foreign_only"] = observe(alice, RANKED, ("body_text:foreignonly", 10))
                report["bare_wildcard"] = observe(alice, RANKED, ("*", 500))
                report["match_all"] = observe(alice, RANKED, ("body_text:alpha OR body_text:beta OR body_text:needle", 500))
                report["owner_count"] = observe(alice, COUNT, ("body_text:needle",))
                report["all_visible"] = observe(alice, "SELECT search_id,user_id,id FROM public.messages ORDER BY search_id")
                report["limit_cases"] = {str(limit): observe(alice, RANKED, ("body_text:needle", limit)) for limit in (1,5,10,20,500)}
                report["explicit_owner"] = observe(alice, RANKED.replace("WHERE search_id", "WHERE user_id='alice' AND search_id"), ("body_text:needle", 20))
                report["contradictory_owner"] = observe(alice, RANKED.replace("WHERE search_id", "WHERE user_id='bob' AND search_id"), ("body_text:needle", 20))
                report["plan_topk"] = observe(alice, "EXPLAIN (ANALYZE,VERBOSE,FORMAT JSON) " + RANKED, ("body_text:needle", 5))
                report["plan_count"] = observe(alice, "EXPLAIN (ANALYZE,VERBOSE,FORMAT JSON) " + COUNT, ("body_text:needle",))
                alice.execute("SET plan_cache_mode=force_generic_plan")
                report["generic_plan_ranked"] = {term: observe(alice, RANKED, (term, 20)) for term in ("body_text:needle", "body_text:foreignonly")}
                report["missing_prepared_helpers"] = {}
                for routine in ("paradedb.with_index(regclass,paradedb.searchqueryinput)",
                                "paradedb.parse_with_field(paradedb.fieldname,text,boolean,boolean)"):
                    admin.execute(sql.SQL("REVOKE EXECUTE ON FUNCTION {} FROM {}").format(sql.SQL(routine), sql.Identifier(roles["alice"])))
                    report["missing_prepared_helpers"][routine] = observe(alice, RANKED, ("body_text:needle", 20))
                    admin.execute(sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(sql.SQL(routine), sql.Identifier(roles["alice"])))
                # ACL denials must also hold if this restricted login changes
                # its advisory session default; role identity remains immutable.
                alice.execute("SET default_transaction_read_only=off")
                report["denied"] = {key: observe(alice, statement) for key, statement in {
                    "withheld": "SELECT withheld FROM public.messages",
                    "write": "DELETE FROM public.messages", "role": "SET ROLE postgres",
                    "create": "CREATE TABLE public.evil(x int)", "temp": "CREATE TEMP TABLE evil(x int)",
                    "sequence": "SELECT nextval('paradedb._typmod_cache_id_seq')",
                }.items()}
                alice.execute("SET app.user_id='bob'")
                report["changed_owner_guc"] = observe(alice, COUNT, ("body_text:foreignonly",))
                alice.execute("SET row_security=off")
                report["disabled_rls"] = observe(alice, RANKED, ("body_text:needle", 10))
            with connect("bob") as bob:
                report["bob_count"] = observe(bob, COUNT, ("body_text:needle",))
                report["bob_ranked"] = observe(bob, RANKED, ("body_text:needle", 5))
            report["admin_topk"] = observe(admin, RANKED, ("body_text:needle", 5))
            report["admin_ahead_of_alice"] = observe(admin, "SELECT count(*) FROM (" + RANKED + ") ranked WHERE user_id='bob'", ("body_text:needle", 200))
            report["role_flags"] = admin.execute("SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls FROM pg_roles WHERE rolname=%s", (roles["alice"],)).fetchone()
            report["effective_extension_functions"] = admin.execute("""SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid)
                FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid AND d.deptype='e'
                JOIN pg_extension e ON e.oid=d.refobjid
                WHERE e.extname='pg_search' AND has_function_privilege(%s,p.oid,'EXECUTE') ORDER BY 1,2,3""", (roles["alice"],)).fetchall()
            assert report["direct_identity"] == (True, True)
            assert report["without_pdb_usage"]["sqlstate"] == "42501"
            assert report["role_flags"] == (False,) * 5
            # The custom scan rewrites the operator; execution ACL on that
            # routine is not a query-admission barrier in this plan.
            assert report["without_operator_grant"]["rows"] == [(10,)]
            assert len(report["score_only_ranked"]["rows"]) == 10
            assert report["without_score_grant"]["sqlstate"] == "42501"
            assert report["operator_only_count"]["rows"] == [(10,)]
            assert report["foreign_only"]["rows"] == []
            assert "rows" in report["match_all"], report["match_all"]
            assert len(report["match_all"]["rows"]) == 12
            assert {row[1] for row in report["match_all"]["rows"]} == {"alice"}
            assert report["owner_count"]["rows"] == [(10,)]
            assert report["bob_count"]["rows"] == [(200,)]
            assert report["admin_ahead_of_alice"]["rows"] == [(200,)]
            assert len(report["all_visible"]["rows"]) == 12
            assert {row[1] for row in report["all_visible"]["rows"]} == {"alice"}
            for limit, result in report["limit_cases"].items():
                assert "rows" in result, (limit, result)
                assert len(result["rows"]) == min(int(limit), 10)
                assert {row[1] for row in result["rows"]} == {"alice"}
                if int(limit) >= 10:
                    assert {row[0] for row in result["rows"]} == set(range(3, 13))
            assert len(report["explicit_owner"]["rows"]) == 10
            assert len(report["generic_plan_ranked"]["body_text:needle"]["rows"]) == 10
            assert report["generic_plan_ranked"]["body_text:foreignonly"]["rows"] == []
            assert all(value.get("sqlstate") == "42501" for value in report["missing_prepared_helpers"].values())
            assert report["contradictory_owner"]["rows"] == []
            assert report["changed_owner_guc"]["rows"] == [(0,)]
            assert report["disabled_rls"]["sqlstate"] == "42501"
            assert all(value.get("sqlstate") == "42501" for value in report["denied"].values())
            assert [row[0] for row in report["before_foreign_corpus"]["rows"]] == [1, 2]
            assert [row[0] for row in report["after_foreign_corpus"]["rows"]] == [2, 1]
            assert report["document_rank_before_foreign"] == report["document_rank_after_foreign"]
            assert [row[0] for row in report["document_rank_after_foreign"]["rows"]] == [1, 2]
            assert report["document_rank_foreign"]["rows"] == []
            assert report["large_tsvector"].get("sqlstate") == "54000"
            local_plan = json.dumps(report["document_rank_plan"]["rows"])
            assert "NormalScanExecState" in local_plan and "TopKScanExecState" not in local_plan
            assert "user_id = 'alice'::text" in local_plan and '"Scores": false' in local_plan
            assert len(report["effective_extension_functions"]) == 4
            for key in ("plan_topk", "plan_count"):
                plan = json.dumps(report[key]["rows"])
                assert "heap_filter" in plan and "user_id = 'alice'::text" in plan
            report["assertions"] = "passed: owner visibility, recall, privileges, direct identity, ranking dependence"
            partial = "CREATE INDEX alice_partial_bm25 ON public.messages USING bm25(search_id,body_text) WITH(key_field='search_id') WHERE user_id='alice'"
            report["partial_with_global"] = observe(admin, partial)
            admin.execute("DROP INDEX public.messages_bm25_idx")
            report["partial_without_global"] = observe(admin, partial)
            if report["partial_without_global"].get("ok"):
                report["second_partial"] = observe(admin, partial.replace("alice", "bob"))
                with connect("alice") as alice:
                    report["partial_ranked"] = observe(alice, RANKED, ("body_text:alpha OR body_text:beta", 10))
            # Separate physical partitions are a different design from partial
            # indexes. Only invented rows are copied for this bounded probe.
            admin.execute("CREATE TABLE public.partitioned_messages (search_id bigint NOT NULL,user_id text NOT NULL,id text,body_text text,PRIMARY KEY(user_id,search_id)) PARTITION BY LIST(user_id)")
            for owner in roles:
                admin.execute(sql.SQL("CREATE TABLE public.{} PARTITION OF public.partitioned_messages FOR VALUES IN ({})").format(sql.Identifier("partition_" + owner), sql.Literal(owner)))
                admin.execute(sql.SQL("CREATE UNIQUE INDEX ON public.{}(search_id)").format(sql.Identifier("partition_" + owner)))
                admin.execute(sql.SQL("CREATE INDEX ON public.{} USING bm25(search_id,body_text) WITH(key_field='search_id')").format(sql.Identifier("partition_" + owner)))
            admin.execute("INSERT INTO public.partitioned_messages SELECT search_id,user_id,id,body_text FROM public.messages")
            admin.execute("ALTER TABLE public.partitioned_messages ENABLE ROW LEVEL SECURITY")
            admin.execute("ALTER TABLE public.partitioned_messages FORCE ROW LEVEL SECURITY")
            admin.execute("CREATE POLICY legacy ON public.partitioned_messages USING(true)")
            for owner, role in roles.items():
                admin.execute(sql.SQL("CREATE POLICY {} ON public.partitioned_messages AS RESTRICTIVE TO {} USING(user_id={})").format(sql.Identifier(owner + "_guard"), sql.Identifier(role), sql.Literal(owner)))
                admin.execute(sql.SQL("GRANT SELECT(search_id,user_id,id,body_text) ON public.partitioned_messages TO {}").format(sql.Identifier(role)))
            partition_query = RANKED.replace("public.messages", "public.partitioned_messages")
            with connect("alice") as alice:
                report["physical_partition_ranked"] = observe(alice, partition_query, ("body_text:alpha OR body_text:beta", 10))
                report["physical_partition_plan"] = observe(alice, "EXPLAIN (ANALYZE,VERBOSE,FORMAT JSON) " + partition_query, ("body_text:alpha OR body_text:beta", 10))
                report["physical_partition_foreign"] = observe(alice, partition_query, ("body_text:foreignonly", 10))
                report["physical_partition_direct_child"] = observe(alice, "SELECT * FROM public.partition_bob")
            report["physical_parent_index"] = observe(admin, "CREATE INDEX partitioned_messages_bm25 ON public.partitioned_messages USING bm25(search_id,body_text) WITH(key_field='search_id')")
            with connect("alice") as alice:
                report["physical_parent_index_ranked"] = observe(alice, partition_query, ("body_text:alpha OR body_text:beta", 10))
                report["physical_parent_index_plan"] = observe(alice, "EXPLAIN (ANALYZE,VERBOSE,FORMAT JSON) " + partition_query, ("body_text:alpha OR body_text:beta", 10))
                admin.execute("INSERT INTO public.partitioned_messages SELECT 10000+i,'bob','extra'||i,'beta foreignextra' FROM generate_series(1,200) AS i")
                admin.execute("VACUUM ANALYZE public.partitioned_messages")
                report["physical_partition_after_foreign_change"] = observe(alice, partition_query, ("body_text:alpha OR body_text:beta", 10))
                report["physical_partition_final_foreign"] = observe(alice, partition_query, ("body_text:foreignonly OR body_text:foreignextra", 10))
                report["physical_partition_final_count"] = observe(alice, COUNT.replace("public.messages", "public.partitioned_messages"), ("body_text:needle",))
            assert report["partial_with_global"].get("sqlstate") == "XX000"
            assert report["partial_without_global"] == {"ok": True}
            assert report["second_partial"].get("sqlstate") == "XX000"
            assert report["partial_ranked"] == report["before_foreign_corpus"]
            assert report["physical_parent_index"] == {"ok": True}
            assert report["physical_parent_index_ranked"] == report["before_foreign_corpus"]
            assert report["physical_partition_after_foreign_change"] == report["before_foreign_corpus"]
            assert report["physical_partition_final_foreign"]["rows"] == []
            assert report["physical_partition_final_count"]["rows"] == [(10,)]
            assert report["physical_partition_direct_child"].get("sqlstate") == "42501"
            report["alternative_assertions"] = "passed: partial-index limitation; physical owner partition scores independent of tested foreign changes"
    finally:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH(FORCE)").format(sql.Identifier(name)))
            for role in created_roles:
                admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        report["cleanup"] = "created database and roles removed"
    return report


if __name__ == "__main__":
    print(json.dumps(probe(os.environ["GMS_RANKED_PROBE_DSN"]), indent=2))
