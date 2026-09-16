"""Synthetic provider and PostgreSQL tests for mailbox-specific ingestion state."""
import base64
from pathlib import Path
from unittest.mock import MagicMock

import psycopg
from psycopg.rows import dict_row
import pytest

from test_owner_ingestion import database as database_fixture, ingestion as ingestion_fixture, message
from gmail_search.gmail import client, invite_guard, url_fetcher
from gmail_search.store import queries

database = database_fixture
ingestion = ingestion_fixture


def provider(owner, history=100):
    service = MagicMock()
    raw = {'id': 'collision', 'threadId': 'thread', 'historyId': str(history), 'internalDate': '1767225600000',
           'payload': {'headers': [{'name': 'Subject', 'value': owner}], 'parts': [
               {'mimeType': 'text/plain', 'body': {'data': base64.urlsafe_b64encode(owner.encode()).decode()}},
               {'mimeType': 'text/plain', 'filename': 'doc.txt', 'body': {'attachmentId': 'attachment', 'size': len(owner)}}]}}
    service.users().messages().list().execute.return_value = {'messages': [{'id': 'collision'}]}
    service.users().messages().get().execute.return_value = raw
    service.users().messages().attachments().get().execute.return_value = {'data': base64.urlsafe_b64encode(owner.encode()).decode()}
    service.users().history().list().execute.return_value = {'history': [{'messagesAdded': [{'message': {'id': 'collision'}}]}]}
    def batch(callback):
        result = MagicMock()
        result.execute.side_effect = lambda: callback('collision', raw, None)
        return result
    service.new_batch_http_request.side_effect = batch
    return service


@pytest.fixture
def mailbox(ingestion, database, monkeypatch):
    conn, owners = ingestion
    conn.execute('CREATE TABLE sync_state (key text PRIMARY KEY, value text)')
    conn.execute('ALTER TABLE attachments ADD COLUMN crawl_attempts int DEFAULT 0, ADD COLUMN crawl_last_attempt timestamptz')
    conn.execute('CREATE TABLE crawl_url_state (filename text PRIMARY KEY, status text)')
    connect = lambda _: psycopg.connect(database[0], autocommit=True, row_factory=dict_row)
    monkeypatch.setattr(client, 'get_connection', connect)
    monkeypatch.setattr(url_fetcher, 'get_connection', connect)
    monkeypatch.setattr(client.time, 'sleep', lambda _: None)
    return conn, owners


def test_download_collisions_preserve_bytes_and_history_per_owner(mailbox, tmp_path):
    conn, (alice, bob) = mailbox
    for owner, history in ((alice, 100), (bob, 200)):
        assert client.download_messages(provider(owner, history), tmp_path/'unused', tmp_path, user_id=owner) == 1
        assert client.download_messages(provider(owner, history), tmp_path/'unused', tmp_path, user_id=owner) == 0
    rows = conn.execute("SELECT user_id,raw_path FROM attachments WHERE filename='doc.txt'").fetchall()
    assert len({row['raw_path'] for row in rows}) == 2
    for row in rows:
        assert Path(row['raw_path']).read_text() == row['user_id']
    assert queries.get_sync_state(conn, 'last_history_id') is None
    assert queries.get_sync_state(conn, f'last_history_id:{alice}') == '100'
    assert queries.get_sync_state(conn, f'last_history_id:{bob}') == '200'
    service = provider(alice, 300)
    assert client.sync_new_messages(service, tmp_path/'unused', tmp_path, user_id=alice) == 1
    assert service.users().history().list.call_args.kwargs['startHistoryId'] == '100'
    assert queries.get_sync_state(conn, f'last_history_id:{bob}') == '200'
    assert queries.get_message(conn, 'collision', user_id=bob).subject == bob


def test_foreign_attachment_parent_refused_before_fetch_or_files(mailbox, tmp_path):
    conn, (alice, bob) = mailbox
    queries.upsert_message(conn, message(alice), user_id=alice)
    service = provider(bob)
    service.reset_mock()
    with pytest.raises(ValueError, match='owner'):
        client.ingest_attachment(conn, service, 'collision', {'filename': 'doc', 'mime_type': 'text/plain', 'attachment_id': 'x'}, attachments_dir=tmp_path, max_attachment_size=100, user_id=bob)
    assert not service.mock_calls
    assert list(tmp_path.iterdir()) == []


def test_invitation_cache_and_cli_metadata_do_not_cross_owners(mailbox, monkeypatch):
    from gmail_search.cli import _att_metas_for_message
    from gmail_search.store.models import Attachment
    conn, (alice, bob) = mailbox
    for owner in (alice, bob):
        queries.upsert_message(conn, message(owner), user_id=owner)
    queries.set_crawl_blocked_reason(conn, message_id='collision', reason='blocked', user_id=alice)
    monkeypatch.setattr(invite_guard, 'should_skip_all_link_crawl', lambda *args: (False, None))
    assert invite_guard.skip_link_crawl_cached(conn, message(bob), [], user_id=alice)
    assert not invite_guard.skip_link_crawl_cached(conn, message(bob), [], user_id=bob)
    queries.upsert_attachment(conn, Attachment(None, 'collision', 'calendar.ics', 'text/calendar', 1), user_id=alice)
    assert _att_metas_for_message(conn, 'collision', user_id=bob) == []


def test_global_crawler_fills_one_representative_for_each_owner(mailbox, tmp_path):
    conn, owners = mailbox
    for owner in owners:
        queries.upsert_message(conn, message(owner), user_id=owner)
        queries.upsert_url_stub(conn, message_id='collision', url='https://example.test/page', user_id=owner)
    aid = conn.execute('SELECT min(id) AS id FROM attachments').fetchone()['id']
    url_fetcher._write_result_sync(tmp_path/'unused', {'id': aid, 'filename': 'URL: https://example.test/page', 'url': 'https://example.test/page'}, 'Page', 'public body')
    assert conn.execute('SELECT count(*) AS n FROM attachments WHERE extracted_text=%s', ('public body',)).fetchone()['n'] == 2


def test_existing_owner_checkpoint_is_seeded_without_foreign_history(mailbox, tmp_path):
    conn, (alice, bob) = mailbox
    queries.upsert_message(conn, message(alice), user_id=alice)
    queries.upsert_message(conn, message(bob), user_id=bob)
    conn.execute('UPDATE messages SET history_id=900 WHERE user_id=%s', (bob,))
    queries.set_sync_state(conn, 'last_history_id', '999')
    assert client.download_messages(provider(alice), tmp_path/'unused', tmp_path, user_id=alice) == 0
    assert queries.get_sync_state(conn, f'last_history_id:{alice}') == '1'
    queries.set_sync_state(conn, f'last_history_id:{alice}', '500')
    assert client.download_messages(provider(alice), tmp_path/'unused', tmp_path, user_id=alice) == 0
    assert queries.get_sync_state(conn, f'last_history_id:{alice}') == '500'


def test_cli_extraction_dispatches_only_selected_owner(mailbox, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gmail_search.cli import _extract_pending_attachments
    from gmail_search.store.models import Attachment
    conn, (alice, bob) = mailbox
    for owner in (alice, bob):
        queries.upsert_message(conn, message(owner), user_id=owner)
        raw = tmp_path / owner
        raw.write_text(owner)
        queries.upsert_attachment(conn, Attachment(None, 'collision', 'doc', 'text/plain', 1, raw_path=str(raw)), user_id=owner)
    monkeypatch.setattr('gmail_search.extract.dispatch', lambda mime, path, config: SimpleNamespace(text=path.read_text(), images=[]))
    assert _extract_pending_attachments(conn, {}, user_id=bob) == 1
    rows = {r['user_id']: r['extracted_text'] for r in conn.execute('SELECT user_id, extracted_text FROM attachments')}
    assert rows == {alice: None, bob: bob}
