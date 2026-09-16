"""Analytical language boundary: compile trusted SQL, never forward user text."""
from dataclasses import FrozenInstanceError

import pytest

from gmail_search.gateway.analytics import QueryRejected, compile_query
from gmail_search.gateway.schema import ANALYTICAL_SCHEMA


def test_parameterizes_literals_and_qualifies_relations():
    result = compile_query("SELECT m.id, m.subject FROM messages m WHERE m.subject = 'private' ORDER BY m.id LIMIT 10")
    assert '"public"."messages"' in result.sql
    assert 'OPERATOR(pg_catalog.=)' in result.sql
    assert 'private' not in result.sql
    assert result.params == ('private', 10)
    assert result.columns == ('id', 'subject')
    with pytest.raises(FrozenInstanceError):
        result.sql = 'SELECT 1'


def test_schema_is_closed_and_immutable():
    assert ANALYTICAL_SCHEMA['messages']['date'] == 'text'
    assert ANALYTICAL_SCHEMA['attachments']['id'] == 'int8'
    assert 'users' not in ANALYTICAL_SCHEMA
    for columns in ANALYTICAL_SCHEMA.values():
        assert not {'raw_json', 'body_html', 'raw_path', 'image_path', 'embedding'} & columns.keys()
    with pytest.raises(TypeError):
        ANALYTICAL_SCHEMA['messages']['id'] = 'evil'


@pytest.mark.parametrize('query', [
    'SELECT count(*) AS total FROM messages',
    'SELECT m.from_addr, count(a.id) AS total FROM messages m LEFT JOIN attachments a ON a.message_id = m.id GROUP BY m.from_addr HAVING count(a.id) > 0 ORDER BY total DESC',
    'WITH senders AS (SELECT from_addr, count(*) AS total FROM messages GROUP BY from_addr) SELECT from_addr, total FROM senders WHERE total > 1',
    'SELECT x.id FROM (SELECT id FROM messages WHERE subject IS NOT NULL) x',
    'SELECT m.id FROM messages m WHERE EXISTS (SELECT a.id FROM attachments a WHERE a.message_id = m.id)',
    'SELECT id, row_number() OVER (PARTITION BY from_addr ORDER BY date DESC) AS rank FROM messages',
    "SELECT lower(from_addr) AS sender, sum(size_bytes) AS bytes FROM messages m JOIN attachments a ON m.id = a.message_id GROUP BY lower(from_addr)",
    "SELECT CASE WHEN size_bytes > 0 THEN size_bytes ELSE 0 END AS size FROM attachments",
    "SELECT coalesce(extracted_text, '') AS text FROM attachments WHERE filename LIKE '%.pdf'",
    "SELECT id FROM messages WHERE id IN ('a', 'b') AND NOT (subject = '')",
    'SELECT id FROM messages WHERE id IN (SELECT message_id FROM attachments)',
    'SELECT CAST(size_bytes AS numeric) AS bytes FROM attachments',
    'SELECT DISTINCT from_addr FROM messages ORDER BY from_addr',
    'SELECT m.* FROM messages m',
])
def test_supported_language(query):
    result = compile_query(query)
    assert result.sql.startswith(('SELECT ', 'WITH '))
    assert result.columns


@pytest.mark.parametrize('query', [
    '', 'SELECT 1; SELECT 2', 'DELETE FROM messages',
    'WITH x AS (DELETE FROM messages RETURNING id) SELECT id FROM x',
    'SELECT * INTO stolen FROM messages', 'SELECT * FROM messages FOR UPDATE',
    'SELECT * FROM pg_catalog.pg_authid', 'SELECT * FROM users',
    'SELECT raw_json FROM messages', 'SELECT image_path FROM attachments',
    'SELECT set_config(\'app.user_id\', \'other\', false)',
    "SELECT pg_read_file('/etc/passwd')", "SELECT pg_sleep(10)",
    'SELECT public.lower(subject) FROM messages',
    'SELECT subject::regclass FROM messages',
    'SELECT subject OPERATOR(public.=) subject FROM messages',
    'SELECT id FROM messages m JOIN attachments a ON m.id = a.message_id',
    'SELECT missing FROM messages', 'SELECT m.missing FROM messages m',
    'SELECT m.id FROM messages m, messages m',
    'SELECT id AS x, subject AS x FROM messages',
    'WITH RECURSIVE x AS (SELECT id FROM messages) SELECT id FROM x',
    'SELECT * FROM generate_series(1, 1000000)',
    'SELECT * FROM messages TABLESAMPLE SYSTEM (1)',
    'SELECT subject COLLATE "C" FROM messages',
    'SELECT $1 FROM messages', 'SELECT subject::public.text FROM messages',
    'SELECT * FROM messages UNION SELECT * FROM messages',
    'SELECT 1 AS "a b"',
])
def test_rejects_unsupported_or_unsafe_sql(query):
    with pytest.raises(QueryRejected):
        compile_query(query)


def test_comments_are_parsed_but_never_forwarded():
    result = compile_query("SELECT id /* secret marker */ FROM messages -- end\n")
    assert 'secret marker' not in result.sql
    assert '--' not in result.sql


def test_size_and_depth_bounds():
    with pytest.raises(QueryRejected):
        compile_query('SELECT ' + ' ' * 100_000 + '1')
    with pytest.raises(QueryRejected):
        compile_query('SELECT ' + 'lower(' * 100 + "'x'" + ')' * 100)


@pytest.mark.parametrize('query', ['SELECT confidential_secret FROM', 'SELECT \ud800'])
def test_parse_failures_are_sanitized(query):
    with pytest.raises(QueryRejected) as caught:
        compile_query(query)
    assert 'confidential_secret' not in str(caught.value)


def test_parameters_follow_sql_order_across_nested_selects():
    result = compile_query("WITH c AS (SELECT 'cte' AS value) SELECT 'target' AS first, q.value FROM (SELECT 'subquery' AS value) q WHERE q.value = 'where' LIMIT 3 OFFSET 1")
    assert result.params == ('cte', 'target', 'subquery', 'where', 3, 1)
    assert result.sql.count('%s') == len(result.params)


def test_qualifies_modulus_without_repeating_in_expression_parameters():
    result = compile_query("SELECT size_bytes % 10 AS tail FROM attachments WHERE lower('X') IN ('x', 'y')")
    assert 'OPERATOR(pg_catalog.%%)' in result.sql
    assert result.params == (10, 'X', 'x', 'y')
    assert 'OPERATOR(pg_catalog.=) ANY (ARRAY[' in result.sql


def test_nested_in_has_linear_output_size():
    expression = 'true'
    for _ in range(6):
        expression = f'({expression} IN (true, false, true, false))'
    result = compile_query('SELECT ' + expression + ' AS accepted')
    assert len(result.params) == 25
    assert len(result.sql) < 2000


def test_null_parameter_keeps_contextual_type_for_numeric_coalesce():
    result = compile_query('SELECT coalesce(size_bytes, NULL) AS size FROM attachments')
    assert result.params == (None,)
    assert 'pg_catalog.text' not in result.sql


@pytest.mark.parametrize('query', [
    'SELECT substr(body_text, 2, 5) AS body_text FROM messages',
    'SELECT pg_catalog.substr(body_text, 1, 100000) AS body_text FROM messages',
])
def test_compiles_bounded_postgres_text_substrings(query):
    result = compile_query(query)
    assert 'pg_catalog."substr"' in result.sql
    assert result.params[-2:] in ((2, 5), (1, 100000))


@pytest.mark.parametrize('query', [
    'SELECT substr(body_text, 0, 1) FROM messages',
    'SELECT substr(body_text, 2147483648, 1) FROM messages',
    'SELECT substr(body_text, 1, 0) FROM messages',
    'SELECT substr(body_text, 1, 100001) FROM messages',
    'SELECT substr(body_text, 1, body_total_chars) FROM messages',
    'SELECT substr(body_text, 1 + 1, 10) FROM messages',
    "SELECT substr(body_text, 1, '10'::int) FROM messages",
    'SELECT substr(body_text, 1) FROM messages',
    'SELECT substr(body_text, 1, 2, 3) FROM messages',
    'SELECT public.substr(body_text, 1, 2) FROM messages',
])
def test_rejects_unbounded_or_ambiguous_substring_arguments(query):
    with pytest.raises(QueryRejected):
        compile_query(query)


@pytest.mark.parametrize('query', [
    'SELECT count(*) FILTER (WHERE size_bytes > 0) AS total FROM attachments',
    'SELECT count(DISTINCT from_addr) AS senders FROM messages',
    'SELECT (SELECT count(*) FROM attachments) AS total',
    'SELECT pg_catalog.lower(subject) AS subject FROM public.messages',
    'SELECT -size_bytes AS negative FROM attachments',
    'SELECT sum(size_bytes) OVER () AS total FROM attachments',
])
def test_additional_supported_forms(query):
    assert compile_query(query).columns


@pytest.mark.parametrize('query', [
    'SELECT 1 AS "' + 'x' * 64 + '"',
    'SELECT id FROM messages ORDER BY subject USING OPERATOR(public.<)',
    'SELECT sum(size_bytes) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM attachments',
    'WITH a(id) AS (SELECT id FROM messages) SELECT id FROM a',
    'SELECT * FROM messages m JOIN attachments a USING (user_id)',
    'SELECT count(*) FROM messages GROUP BY ROLLUP(from_addr)',
    'SELECT ARRAY(SELECT id FROM messages)',
    'SELECT subject::varchar(10) FROM messages',
    'SELECT subject::text[] FROM messages',
    'SELECT id FROM ONLY messages',
    'SELECT id FROM messages LIMIT -1',
    'SELECT id FROM messages OFFSET 1000001',
    'SELECT lower(VARIADIC labels) FROM messages',
    'SELECT * FROM messages m, LATERAL (SELECT m.id) x',
    'SELECT lower(subject) OVER () FROM messages',
])
def test_rejects_unimplemented_ast_modifiers(query):
    with pytest.raises(QueryRejected):
        compile_query(query)
def test_text_amplification_operator_is_not_in_public_language():
    import pytest
    from gmail_search.gateway.analytics import compile_query, QueryRejected
    with pytest.raises(QueryRejected):
        compile_query("WITH x AS (SELECT 'data' AS v) SELECT v || v AS doubled FROM x")


def test_rejects_hundreds_of_cross_join_relations():
    query = 'SELECT count(*) FROM ' + ', '.join(f'messages m{i}' for i in range(300))
    with pytest.raises(QueryRejected):
        compile_query(query)


def test_relation_complexity_budget_boundary():
    def query(count):
        return 'SELECT count(*) FROM ' + ', '.join(f'messages m{i}' for i in range(count))
    assert compile_query(query(16)).columns == ('count',)
    with pytest.raises(QueryRejected):
        compile_query(query(17))


def test_select_complexity_budget_counts_all_ctes():
    def query(count):
        return 'WITH ' + ', '.join(f'c{i} AS (SELECT {i} AS n)' for i in range(count)) + ' SELECT n FROM c0'
    assert compile_query(query(15)).columns == ('n',)
    with pytest.raises(QueryRejected):
        compile_query(query(16))


def test_join_complexity_budget_boundary():
    def query(count):
        # Mix relations and derived tables so neither of their separate limits
        # rejects the input first; this exercises the join limit independently.
        sources = [f'messages m{i}' if i % 2 else f'(SELECT 1 AS n) q{i}'
                   for i in range(count + 1)]
        return 'SELECT count(*) FROM ' + ' CROSS JOIN '.join(sources)
    assert compile_query(query(15)).columns == ('count',)
    with pytest.raises(QueryRejected):
        compile_query(query(16))
