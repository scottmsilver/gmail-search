"""Crawler operations must retain row identity after IDs become owner-local."""
import pytest

from gmail_search.gmail import url_fetcher as uf
from gmail_search.store.db import get_connection
from gmail_search.store import queries
from test_url_fetcher_db import db as db_fixture, _seed, STUB, URL


db = db_fixture
OTHER = 'URL: https://other.example.test/page'


@pytest.fixture
def colliding(db):
    _seed(db, [('uA', 'a1', STUB), ('uA', 'a2', STUB),
               ('uB', 'b1', STUB), ('uB', 'b2', STUB), ('uC', 'c1', OTHER)])
    conn = get_connection(db)
    try:
        conn.execute('ALTER TABLE attachments DROP CONSTRAINT attachments_pkey')
        conn.execute('ALTER TABLE attachments ADD PRIMARY KEY(user_id,id)')
        for owner, mid, aid in [('uA','a1',101), ('uA','a2',102),
                                ('uB','b1',102), ('uB','b2',103), ('uC','c1',102)]:
            conn.execute('UPDATE attachments SET id=%s WHERE user_id=%s AND message_id=%s', (aid, owner, mid))
        conn.commit()
    finally:
        conn.close()
    return db


def rows(db):
    conn = get_connection(db)
    try:
        return {(r['user_id'],r['id']):dict(r) for r in conn.execute(
            'SELECT user_id,id,extracted_text,crawl_attempts FROM attachments').fetchall()}
    finally:
        conn.close()


@pytest.mark.parametrize('operation', ['attempt', 'abandon'])
def test_url_bookkeeping_does_not_touch_another_url_with_same_id(colliding, operation):
    if operation == 'attempt':
        uf._mark_attempt_sync(colliding, STUB)
    else:
        uf._abandon_sync(colliding, STUB)
    actual = rows(colliding)
    expected = 1 if operation == 'attempt' else queries._MAX_CRAWL_ATTEMPTS
    assert all(row['crawl_attempts'] == expected for (owner, _), row in actual.items() if owner != 'uC')
    assert actual[('uC',102)]['crawl_attempts'] == 0


def test_representative_and_duplicate_ids_include_owner(colliding):
    uf._write_result_sync(colliding, {'id':102,'user_id':'uB','url':URL,'filename':STUB}, 'Title', 'BODY')
    actual = rows(colliding)
    assert actual[('uA',101)]['extracted_text'] == 'BODY'
    assert actual[('uB',102)]['extracted_text'] == 'BODY'
    assert actual[('uA',102)]['crawl_attempts'] == queries._MAX_CRAWL_ATTEMPTS
    assert actual[('uB',103)]['crawl_attempts'] == queries._MAX_CRAWL_ATTEMPTS
    assert actual[('uC',102)]['crawl_attempts'] == 0
    assert actual[('uC',102)]['extracted_text'] is None


def test_denied_url_purge_preserves_foreign_same_id(colliding, monkeypatch):
    monkeypatch.setattr('gmail_search.gmail.url_extract._is_denied', lambda url: url == URL)
    conn = get_connection(colliding)
    try:
        pending = queries.pending_url_stubs(conn, 20)
        assert len(pending) == 1
        assert pending[0]['user_id'] == 'uC'
    finally:
        conn.close()
    assert set(rows(colliding)) == {('uC',102)}


def test_ownerless_colliding_stub_is_refused_without_changes(colliding):
    before = rows(colliding)
    with pytest.raises(ValueError, match='owner-qualified'):
        uf._write_result_sync(colliding, {'id':102,'url':URL,'filename':STUB}, 'Title', 'BODY')
    assert rows(colliding) == before
