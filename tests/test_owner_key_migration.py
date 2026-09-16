"""Positive synthetic ingestion regression gates for the owner-qualified schema."""
from datetime import datetime, timezone

import psycopg
import pytest

from test_gateway_database_integration import database as database_fixture
from gmail_search.store.models import Attachment, Message
from gmail_search.store import queries

from test_owner_ingestion import ingestion as ingestion_fixture

database = database_fixture
ingestion = ingestion_fixture


def test_colliding_message_id_preserves_both_owners(ingestion):
    conn, (alice, bob) = ingestion
    def message(subject):
        return Message('collision', 'thread', 'sender@example.test', 'to@example.test', subject,
                       subject, '', datetime(2026, 1, 1, tzinfo=timezone.utc), [], 1, '{}')
    queries.upsert_message(conn, message('Alice private'), user_id=alice)
    queries.upsert_message(conn, message('Bob private'), user_id=bob)
    actual = {row['user_id']: row['subject'] for row in conn.execute("SELECT user_id,subject FROM public.messages WHERE id='collision'")}
    assert actual == {alice: 'Alice private', bob: 'Bob private'}


def test_foreign_owner_attachment_cannot_replace_existing_path(ingestion):
    conn, (alice, bob) = ingestion
    conn.execute("INSERT INTO public.messages (id,user_id) VALUES ('collision',%s)", (alice,))
    original = Attachment(None, 'collision', 'document.pdf', 'application/pdf', 10, raw_path='/synthetic/alice.pdf')
    foreign = Attachment(None, 'collision', 'document.pdf', 'application/pdf', 20, raw_path='/synthetic/bob.pdf')
    queries.upsert_attachment(conn, original, user_id=alice)
    # Bob has no matching parent message; a composite FK must reject the write.
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        queries.upsert_attachment(conn, foreign, user_id=bob)
    row = conn.execute("SELECT user_id,raw_path FROM public.attachments WHERE message_id='collision'").fetchone()
    assert row == {'user_id': alice, 'raw_path': '/synthetic/alice.pdf'}
