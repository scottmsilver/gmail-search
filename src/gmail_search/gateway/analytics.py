"""Compile a deliberately small analytical SQL language into fresh SQL.

No deparser or caller SQL fragment is used. Every AST node and field is checked.
Only native mailbox columns, pg_catalog functions/types/operators, SELECT joins,
grouping, anonymous default-frame windows, CTEs and subqueries are supported.
Execution limits, owner authorization and RLS are independently enforced by the
database gateway. This module performs no database or filesystem operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import re

from pglast.parser import ParseError, parse_sql_json

from .schema import ANALYTICAL_SCHEMA

MAX_INPUT_BYTES = 32_768
MAX_AST_DEPTH = 64
MAX_AST_NODES = 8_000
MAX_OUTPUT_COLUMNS = 128
MAX_RELATIONS = 16
MAX_SELECTS = 16
MAX_JOINS = 15
_PG_INT32_MAX = 2_147_483_647
_MAX_SUBSTR_LENGTH = 100_000
# Below PostgreSQL's 63-byte truncation boundary, so silently truncated aliases
# cannot become accepted identifiers.
_IDENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]{0,47}\Z', re.ASCII)
_PARAM = re.compile('\x01([0-9]+)\x02')
_TYPES = frozenset({'text', 'bool', 'int2', 'int4', 'int8', 'float4', 'float8', 'numeric', 'date', 'timestamp', 'timestamptz', 'interval'})
_OPERATORS = frozenset({'=', '<>', '<', '>', '<=', '>=', '+', '-', '*', '/', '%', '~~', '!~~', '~~*', '!~~*'})
# Minimum/maximum argument counts. Omitted routines cannot be invoked indirectly.
_FUNCTIONS = {
    'count': (0, 1), 'sum': (1, 1), 'avg': (1, 1), 'min': (1, 1), 'max': (1, 1),
    'lower': (1, 1), 'upper': (1, 1), 'length': (1, 1), 'btrim': (1, 2),
    'abs': (1, 1), 'round': (1, 2), 'ceil': (1, 1), 'floor': (1, 1),
    'date_trunc': (2, 2), 'date_part': (2, 2),
    'row_number': (0, 0), 'rank': (0, 0), 'dense_rank': (0, 0),
    'lag': (1, 3), 'lead': (1, 3), 'first_value': (1, 1), 'last_value': (1, 1),
}
_AGGREGATES = frozenset({'count', 'sum', 'avg', 'min', 'max'})
_WINDOWS = frozenset({'row_number', 'rank', 'dense_rank', 'lag', 'lead', 'first_value', 'last_value'})


class QueryRejected(ValueError):
    """Unsupported query; messages contain no input SQL or bound data."""


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    params: tuple
    columns: tuple[str, ...]


def _reject():
    raise QueryRejected('Query uses unsupported or ambiguous analytical SQL.')


def _fields(value, allowed):
    if not isinstance(value, dict) or set(value) - (set(allowed.split()) | {'location'}):
        _reject()
    return value


def _node(value, name=None):
    if not isinstance(value, dict) or len(value) != 1:
        _reject()
    kind, body = next(iter(value.items()))
    if name is not None and kind != name:
        _reject()
    return body if name else (kind, body)


def _strings(values):
    result = []
    for value in values:
        body = _fields(_node(value, 'String'), 'sval')
        result.append(body.get('sval', ''))
    return result


def _ident(value):
    if not isinstance(value, str) or not _IDENT.fullmatch(value):
        _reject()
    return '"' + value + '"'


def _builtin(values, allowed):
    names = _strings(values)
    if len(names) == 2 and names[0] == 'pg_catalog':
        names = names[1:]
    if len(names) != 1 or names[0] not in allowed:
        _reject()
    return names[0]


class _Scope:
    def __init__(self, parent=None):
        self.relations = {}
        self.parent = parent

    def add(self, alias, columns):
        _ident(alias)
        if alias in self.relations:
            _reject()
        self.relations[alias] = tuple(columns)

    def column(self, parts):
        if len(parts) == 2:
            alias, column = parts
            if alias in self.relations:
                if column not in self.relations[alias]:
                    _reject()
                return _ident(alias) + '.' + _ident(column)
        elif len(parts) == 1:
            column = parts[0]
            matches = [alias for alias, columns in self.relations.items() if column in columns]
            if len(matches) > 1:
                _reject()
            if matches:
                return _ident(matches[0]) + '.' + _ident(column)
        else:
            _reject()
        if self.parent:
            return self.parent.column(parts)
        _reject()


class _Compiler:
    def __init__(self):
        self.params = []

    def parameter(self, value, kind):
        index = len(self.params)
        self.params.append(value)
        if value is None:
            # NULL must take its surrounding native expression's type (for
            # example COALESCE(int8, NULL)); forcing text changes SQL semantics.
            return f'\x01{index}\x02'
        return f'CAST(\x01{index}\x02 AS pg_catalog.{kind})'

    def bounded_int_constant(self, value, minimum, maximum):
        constant = _fields(_node(value, 'A_Const'), 'ival')
        integer = constant.get('ival')
        if not isinstance(integer, dict) or set(integer) != {'ival'}:
            _reject()
        number = integer['ival']
        if type(number) is not int or not minimum <= number <= maximum:
            _reject()
        return self.parameter(number, 'int4')

    def select(self, value, ctes=None, parent=None):
        stmt = _fields(_node(value, 'SelectStmt'), 'targetList fromClause whereClause groupClause havingClause sortClause limitCount limitOffset limitOption op withClause distinctClause')
        if stmt.get('op', 'SETOP_NONE') != 'SETOP_NONE' or stmt.get('limitOption') not in (None, 'LIMIT_OPTION_DEFAULT', 'LIMIT_OPTION_COUNT'):
            _reject()
        ctes = dict(ctes or {})
        prefix = ''
        if 'withClause' in stmt:
            clause = _fields(stmt['withClause'], 'ctes')
            fragments = []
            local_names = set()
            for item in clause.get('ctes', []):
                cte = _fields(_node(item, 'CommonTableExpr'), 'ctename ctematerialized ctequery')
                name = cte['ctename']
                if name in local_names or cte.get('ctematerialized') != 'CTEMaterializeDefault':
                    _reject()
                local_names.add(name)
                query, columns = self.select(cte['ctequery'], ctes, parent)
                ctes[name] = columns
                fragments.append(_ident(name) + ' AS (' + query + ')')
            if not fragments:
                _reject()
            prefix = 'WITH ' + ', '.join(fragments) + ' '
        scope = _Scope(parent)
        relations = [self.relation(item, scope, ctes) for item in stmt.get('fromClause', [])]
        targets, columns = [], []
        for item in stmt.get('targetList', []):
            target = _fields(_node(item, 'ResTarget'), 'name val')
            value = target['val']
            kind, body = _node(value)
            if kind == 'ColumnRef' and body.get('fields') and 'A_Star' in body['fields'][-1]:
                _fields(body, 'fields')
                if 'name' in target:
                    _reject()
                parts = _strings(body['fields'][:-1])
                if len(parts) > 1:
                    _reject()
                aliases = parts or list(scope.relations)
                if not aliases:
                    _reject()
                for alias in aliases:
                    if alias not in scope.relations:
                        _reject()
                    for column in scope.relations[alias]:
                        columns.append(column)
                        targets.append(scope.column([alias, column]) + ' AS ' + _ident(column))
            else:
                emitted = self.expr(value, scope, ctes)
                name = target.get('name')
                if not name:
                    if kind == 'ColumnRef':
                        name = _strings(body['fields'])[-1]
                    elif kind == 'FuncCall':
                        name = _strings(body['funcname'])[-1]
                    else:
                        name = f'column_{len(columns) + 1}'
                targets.append(emitted + ' AS ' + _ident(name))
                columns.append(name)
        if not columns or len(columns) > MAX_OUTPUT_COLUMNS or len(set(columns)) != len(columns):
            _reject()
        distinct = ''
        if 'distinctClause' in stmt:
            if stmt['distinctClause'] != [{}]:
                _reject()
            distinct = 'DISTINCT '
        sql = prefix + 'SELECT ' + distinct + ', '.join(targets)
        if relations:
            sql += ' FROM ' + ', '.join(relations)
        if 'whereClause' in stmt:
            sql += ' WHERE ' + self.expr(stmt['whereClause'], scope, ctes)
        if 'groupClause' in stmt:
            sql += ' GROUP BY ' + ', '.join(self.expr(x, scope, ctes) for x in stmt['groupClause'])
        if 'havingClause' in stmt:
            sql += ' HAVING ' + self.expr(stmt['havingClause'], scope, ctes)
        if 'sortClause' in stmt:
            sql += ' ORDER BY ' + self.order(stmt['sortClause'], scope, ctes, columns)
        for key, word in [('limitCount', ' LIMIT '), ('limitOffset', ' OFFSET ')]:
            if key in stmt:
                const = _fields(_node(stmt[key], 'A_Const'), 'ival')
                number = const.get('ival', {}).get('ival', 0)
                if not isinstance(number, int) or not 0 <= number <= 1_000_000:
                    _reject()
                sql += word + self.parameter(number, 'int8')
        return sql, tuple(columns)

    def relation(self, value, scope, ctes):
        kind, body = _node(value)
        if kind == 'RangeVar':
            _fields(body, 'relname schemaname inh relpersistence alias')
            if body.get('schemaname') not in (None, 'public') or body.get('inh') is not True or body.get('relpersistence') != 'p':
                _reject()
            name = body['relname']
            alias = name
            if 'alias' in body:
                alias = _fields(body['alias'], 'aliasname')['aliasname']
            if name in ctes and 'schemaname' not in body:
                columns, source = ctes[name], _ident(name)
            elif name in ANALYTICAL_SCHEMA:
                columns, source = ANALYTICAL_SCHEMA[name], '"public".' + _ident(name)
            else:
                _reject()
            scope.add(alias, columns)
            return source + ' AS ' + _ident(alias)
        if kind == 'RangeSubselect':
            _fields(body, 'subquery alias')
            alias = _fields(body.get('alias'), 'aliasname')['aliasname']
            query, columns = self.select(body['subquery'], ctes)
            scope.add(alias, columns)
            return '(' + query + ') AS ' + _ident(alias)
        if kind == 'JoinExpr':
            _fields(body, 'jointype larg rarg quals')
            joins = {'JOIN_INNER': 'INNER JOIN', 'JOIN_LEFT': 'LEFT JOIN', 'JOIN_RIGHT': 'RIGHT JOIN', 'JOIN_FULL': 'FULL JOIN'}
            join = joins.get(body.get('jointype'))
            if not join:
                _reject()
            left = self.relation(body['larg'], scope, ctes)
            right = self.relation(body['rarg'], scope, ctes)
            if 'quals' not in body:
                if body['jointype'] != 'JOIN_INNER':
                    _reject()
                return '(' + left + ' CROSS JOIN ' + right + ')'
            return '(' + left + ' ' + join + ' ' + right + ' ON ' + self.expr(body['quals'], scope, ctes) + ')'
        _reject()

    def order(self, values, scope, ctes, aliases=()):
        results = []
        for value in values:
            body = _fields(_node(value, 'SortBy'), 'node sortby_dir sortby_nulls')
            kind, node = _node(body['node'])
            parts = _strings(node['fields']) if kind == 'ColumnRef' else []
            expr = _ident(parts[0]) if len(parts) == 1 and parts[0] in aliases else self.expr(body['node'], scope, ctes)
            direction = {'SORTBY_DEFAULT': '', 'SORTBY_ASC': ' ASC', 'SORTBY_DESC': ' DESC'}.get(body.get('sortby_dir'))
            nulls = {'SORTBY_NULLS_DEFAULT': '', 'SORTBY_NULLS_FIRST': ' NULLS FIRST', 'SORTBY_NULLS_LAST': ' NULLS LAST'}.get(body.get('sortby_nulls'))
            if direction is None or nulls is None:
                _reject()
            results.append(expr + direction + nulls)
        return ', '.join(results)

    def expr(self, value, scope, ctes):
        kind, body = _node(value)
        if kind == 'ColumnRef':
            _fields(body, 'fields')
            return scope.column(_strings(body['fields']))
        if kind == 'A_Const':
            _fields(body, 'ival fval sval boolval isnull')
            keys = set(body) - {'location'}
            if len(keys) != 1:
                _reject()
            if body.get('isnull') is True:
                return self.parameter(None, 'text')
            if 'ival' in body:
                number = body['ival'].get('ival', 0)
                return self.parameter(number, 'int8')
            if 'fval' in body:
                return self.parameter(Decimal(body['fval']['fval']), 'numeric')
            if 'sval' in body:
                return self.parameter(body['sval'].get('sval', ''), 'text')
            if 'boolval' in body:
                return self.parameter(body['boolval'].get('boolval', False), 'bool')
            _reject()
        if kind == 'A_Expr':
            _fields(body, 'kind name lexpr rexpr')
            operator = _builtin(body['name'], _OPERATORS)
            mode = body['kind']
            if mode == 'AEXPR_IN':
                if operator not in ('=', '<>'):
                    _reject()
                items = _fields(_node(body['rexpr'], 'List'), 'items')['items']
                left = self.expr(body['lexpr'], scope, ctes)
                if not items:
                    _reject()
                # Evaluate the left expression once. Repeating it for every item
                # makes nested IN expressions expand exponentially at compile time.
                quantifier = 'ANY' if operator == '=' else 'ALL'
                elements = ', '.join(self.expr(item, scope, ctes) for item in items)
                return '(' + left + ' OPERATOR(pg_catalog.' + operator + ') ' + quantifier + ' (ARRAY[' + elements + ']))'
            if mode not in ('AEXPR_OP', 'AEXPR_LIKE', 'AEXPR_ILIKE'):
                _reject()
            right = self.expr(body['rexpr'], scope, ctes)
            if 'lexpr' not in body:
                if operator not in ('+', '-'):
                    _reject()
                return '(OPERATOR(pg_catalog.' + operator + ') ' + right + ')'
            left = self.expr(body['lexpr'], scope, ctes)
            return '(' + left + ' OPERATOR(pg_catalog.' + operator + ') ' + right + ')'
        if kind == 'BoolExpr':
            _fields(body, 'boolop args')
            args = [self.expr(arg, scope, ctes) for arg in body['args']]
            mode = body['boolop']
            if mode == 'NOT_EXPR' and len(args) == 1:
                return '(NOT ' + args[0] + ')'
            if mode not in ('AND_EXPR', 'OR_EXPR') or len(args) < 2:
                _reject()
            return '(' + (' AND ' if mode == 'AND_EXPR' else ' OR ').join(args) + ')'
        if kind == 'NullTest':
            _fields(body, 'arg nulltesttype')
            suffix = {'IS_NULL': ' IS NULL)', 'IS_NOT_NULL': ' IS NOT NULL)'}.get(body['nulltesttype'])
            if suffix is None:
                _reject()
            return '(' + self.expr(body['arg'], scope, ctes) + suffix
        if kind == 'TypeCast':
            _fields(body, 'arg typeName')
            target = _fields(body['typeName'], 'names typemod')
            if target.get('typemod') != -1:
                _reject()
            name = _builtin(target['names'], _TYPES)
            return 'CAST(' + self.expr(body['arg'], scope, ctes) + ' AS pg_catalog.' + name + ')'
        if kind == 'CoalesceExpr':
            _fields(body, 'args')
            return 'COALESCE(' + ', '.join(self.expr(x, scope, ctes) for x in body['args']) + ')'
        if kind == 'CaseExpr':
            _fields(body, 'arg args defresult')
            result = 'CASE'
            if 'arg' in body:
                result += ' ' + self.expr(body['arg'], scope, ctes)
            for item in body.get('args', []):
                when = _fields(_node(item, 'CaseWhen'), 'expr result')
                result += ' WHEN ' + self.expr(when['expr'], scope, ctes) + ' THEN ' + self.expr(when['result'], scope, ctes)
            if 'defresult' in body:
                result += ' ELSE ' + self.expr(body['defresult'], scope, ctes)
            return '(' + result + ' END)'
        if kind == 'FuncCall':
            _fields(body, 'funcname args agg_star agg_distinct agg_filter over funcformat')
            if body.get('funcformat') != 'COERCE_EXPLICIT_CALL':
                _reject()
            args = body.get('args', [])
            name = _builtin(body['funcname'], set(_FUNCTIONS) | {'substr'})
            if name == 'substr':
                if (len(args) != 3 or body.get('agg_star') or body.get('agg_distinct')
                        or 'agg_filter' in body or 'over' in body):
                    _reject()
                return ('pg_catalog.' + _ident('substr') + '('
                        + self.expr(args[0], scope, ctes) + ', '
                        + self.bounded_int_constant(args[1], 1, _PG_INT32_MAX) + ', '
                        + self.bounded_int_constant(args[2], 1, _MAX_SUBSTR_LENGTH) + ')')
            low, high = _FUNCTIONS[name]
            star = body.get('agg_star', False)
            if not low <= len(args) <= high or (star and (name != 'count' or args)) or (name == 'count' and not star and not args):
                _reject()
            if (body.get('agg_distinct') or 'agg_filter' in body) and name not in _AGGREGATES:
                _reject()
            if name in _WINDOWS and 'over' not in body:
                _reject()
            text = '*' if star else ', '.join(self.expr(x, scope, ctes) for x in args)
            if body.get('agg_distinct'):
                text = 'DISTINCT ' + text
            result = 'pg_catalog.' + _ident(name) + '(' + text + ')'
            if 'agg_filter' in body:
                result += ' FILTER (WHERE ' + self.expr(body['agg_filter'], scope, ctes) + ')'
            if 'over' in body:
                if name not in _AGGREGATES | _WINDOWS:
                    _reject()
                over = _fields(body['over'], 'partitionClause orderClause frameOptions')
                if over.get('frameOptions') != 1058:
                    _reject()
                pieces = []
                if 'partitionClause' in over:
                    pieces.append('PARTITION BY ' + ', '.join(self.expr(x, scope, ctes) for x in over['partitionClause']))
                if 'orderClause' in over:
                    pieces.append('ORDER BY ' + self.order(over['orderClause'], scope, ctes))
                result += ' OVER (' + ' '.join(pieces) + ')'
            return result
        if kind == 'SubLink':
            _fields(body, 'subLinkType testexpr subselect')
            mode = body['subLinkType']
            query, columns = self.select(body['subselect'], ctes, scope)
            if mode == 'EXISTS_SUBLINK' and 'testexpr' not in body:
                return 'EXISTS (' + query + ')'
            if mode == 'EXPR_SUBLINK' and 'testexpr' not in body and len(columns) == 1:
                return '(' + query + ')'
            if mode == 'ANY_SUBLINK' and 'testexpr' in body and len(columns) == 1:
                return '(' + self.expr(body['testexpr'], scope, ctes) + ' OPERATOR(pg_catalog.=) ANY (' + query + '))'
            _reject()
        _reject()


def compile_query(text: str) -> CompiledQuery:
    """Return parameterized SQL; reject unsupported inputs without quoting them."""
    try:
        if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > MAX_INPUT_BYTES or '\x00' in text:
            _reject()
        tree = json.loads(parse_sql_json(text))
        count = 0
        complexity = {'RangeVar': 0, 'SelectStmt': 0, 'JoinExpr': 0}
        limits = {'RangeVar': MAX_RELATIONS, 'SelectStmt': MAX_SELECTS, 'JoinExpr': MAX_JOINS}
        pending = [(tree, 0)]
        while pending:
            value, depth = pending.pop()
            count += 1
            if count > MAX_AST_NODES or depth > MAX_AST_DEPTH:
                _reject()
            if isinstance(value, dict):
                for kind in complexity:
                    if kind in value:
                        complexity[kind] += 1
                        if complexity[kind] > limits[kind]:
                            _reject()
                pending.extend((v, depth + 1) for v in value.values())
            elif isinstance(value, list):
                pending.extend((v, depth + 1) for v in value)
        statements = tree['stmts']
        if len(statements) != 1:
            _reject()
        compiler = _Compiler()
        sql, columns = compiler.select(statements[0]['stmt'])
        ordered = []

        def placeholder(match):
            ordered.append(compiler.params[int(match.group(1))])
            return '%s'

        sql = _PARAM.sub(placeholder, sql.replace('%', '%%'))
        return CompiledQuery(sql, tuple(ordered), columns)
    except QueryRejected:
        raise
    except (ParseError, ValueError, KeyError, TypeError, RecursionError, OverflowError):
        raise QueryRejected('Query is not valid supported analytical SQL.') from None
