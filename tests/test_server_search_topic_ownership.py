"""Search topic joins run against synthetic colliding-owner tables only."""
import inspect
import sqlite3

import pytest

from gmail_search import server
from gmail_search.config import load_config
from gmail_search.search.engine import ThreadMatch, ThreadResult
from gmail_search.store import db


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    path = tmp_path / 'topic-fixture.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE topics (user_id TEXT NOT NULL, topic_id TEXT NOT NULL,
                parent_id TEXT, label TEXT, PRIMARY KEY(user_id, topic_id));
            CREATE TABLE message_topics (user_id TEXT NOT NULL, message_id TEXT NOT NULL,
                topic_id TEXT NOT NULL, PRIMARY KEY(user_id,message_id,topic_id));
            INSERT INTO topics VALUES
                ('alice','shared',NULL,'Alice shared'),
                ('alice','alice-only',NULL,'Alice private'),
                ('alice','parent-in-bob',NULL,'Alice leaf'),
                ('alice','alice-parent',NULL,'Alice parent'),
                ('alice','alice-child','alice-parent','Alice child'),
                ('bob','shared',NULL,'Bob shared'),
                ('bob','bob-only',NULL,'Bob private'),
                ('bob','parent-in-bob',NULL,'Bob parent'),
                ('bob','bob-child','parent-in-bob','Bob child');
            INSERT INTO message_topics VALUES
                ('alice','same','shared'), ('alice','same','alice-only'),
                ('alice','same','parent-in-bob'), ('alice','same','alice-parent'),
                ('bob','same','shared'), ('bob','same','bob-only'),
                ('bob','same','parent-in-bob');
        ''')

    connections = []

    class Connection:
        closed = False

        def __init__(self):
            self.conn = sqlite3.connect(path)
            self.conn.row_factory = sqlite3.Row
            connections.append(self)

        def execute(self, query, params=()):
            # Only parameter spelling differs for these ordinary SELECT joins.
            return self.conn.execute(query.replace('%s', '?'), params)

        def close(self):
            self.closed = True
            self.conn.close()

    class Engine:
        def __init__(self, *args, **kwargs):
            pass

        def search_threads(self, *args, **kwargs):
            return [ThreadResult(thread_id='thread', score=.9, similarity=.8,
                subject='synthetic', participants=[], message_count=1,
                date_first='2026-01-01', date_last='2026-01-01', user_replied=False,
                matches=[ThreadMatch(message_id='same', score=.8, from_addr='synthetic',
                    date='2026-01-01', snippet='synthetic', match_type='semantic')])]

    monkeypatch.setattr(db, 'get_connection', lambda _: Connection())
    monkeypatch.setattr(db, 'reap_stale_jobs', lambda _: None)
    monkeypatch.setattr(server, 'SearchEngine', Engine)
    monkeypatch.setattr('gmail_search.index.searcher.resolve_active_index_dir', lambda *a, **k: tmp_path)
    app = server.create_app(path, tmp_path, load_config(data_dir=tmp_path))
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/api/search')
    return route.endpoint, connections


def invoke(endpoint, owner, *, detail='snippet', facets=True):
    return endpoint(q='synthetic', k=10, sort='relevance', filter=True,
                    date_from=None, date_to=None, match_detail=detail,
                    max_matches=3, include_facets=facets, user_id=owner)


@pytest.mark.parametrize('owner,expected', [
    ('alice', {'shared', 'alice-only', 'parent-in-bob'}),
    ('bob', {'shared', 'bob-only'}),
])
def test_search_topic_ids_are_owner_scoped_even_when_facets_omitted(endpoint, owner, expected):
    call, connections = endpoint
    result = invoke(call, owner, facets=False)
    assert set(result['results'][0]['topic_ids']) == expected
    assert all(conn.closed for conn in connections)


@pytest.mark.parametrize('detail', ['refs', 'snippet'])
@pytest.mark.parametrize('owner,labels', [
    ('alice', {'shared': 'Alice shared', 'alice-only': 'Alice private', 'parent-in-bob': 'Alice leaf'}),
    ('bob', {'shared': 'Bob shared', 'bob-only': 'Bob private'}),
])
def test_search_facets_scope_topic_labels_and_leaf_detection(endpoint, detail, owner, labels):
    call, connections = endpoint
    result = invoke(call, owner, detail=detail)
    assert {item['topic_id']: item['label'] for item in result['facets']} == labels
    assert all(item['count'] == 1 for item in result['facets'])
    assert all(conn.closed for conn in connections)


def test_topic_helper_requires_owner_without_bootstrap_fallback(endpoint):
    call, connections = endpoint
    helper = inspect.getclosurevars(call).nonlocals['_lookup_message_topics']
    before = len(connections)
    for owner in (None, '', 42):
        with pytest.raises(ValueError, match='owner'):
            helper({'same'}, user_id=owner)
    assert len(connections) == before
