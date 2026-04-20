from datetime import datetime

from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import Attachment, EmbeddingRecord, Message
from gmail_search.store.queries import (
    embedding_exists,
    fill_drive_attachment,
    get_attachments_for_message,
    get_message,
    get_messages_without_embeddings,
    get_sync_state,
    insert_embedding,
    load_all_embeddings,
    set_sync_state,
    upsert_attachment,
    upsert_drive_stub,
    upsert_message,
)


def _make_message(id="msg1"):
    return Message(
        id=id,
        thread_id="thread1",
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject="Test subject",
        body_text="Hello world",
        body_html="<p>Hello world</p>",
        date=datetime(2025, 6, 15, 10, 30),
        labels=["INBOX", "IMPORTANT"],
        history_id=12345,
        raw_json='{"id": "msg1"}',
    )


def test_upsert_and_get_message(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    msg = _make_message()
    upsert_message(conn, msg)
    result = get_message(conn, "msg1")
    assert result is not None
    assert result.subject == "Test subject"
    assert result.from_addr == "alice@example.com"
    conn.close()


def test_upsert_message_updates_existing(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    msg = _make_message()
    upsert_message(conn, msg)
    msg.subject = "Updated subject"
    upsert_message(conn, msg)
    result = get_message(conn, "msg1")
    assert result.subject == "Updated subject"
    conn.close()


def test_get_messages_without_embeddings(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message("msg1"))
    upsert_message(conn, _make_message("msg2"))
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id="msg1",
            attachment_id=None,
            chunk_type="message",
            chunk_text="test",
            embedding=b"\x00" * 12,
            model="test-model",
        ),
    )
    result = get_messages_without_embeddings(conn, model="test-model")
    assert len(result) == 1
    assert result[0].id == "msg2"
    conn.close()


def test_attachment_crud(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message())
    att = Attachment(
        id=None,
        message_id="msg1",
        filename="doc.pdf",
        mime_type="application/pdf",
        size_bytes=1024,
    )
    att_id = upsert_attachment(conn, att)
    assert att_id is not None
    atts = get_attachments_for_message(conn, "msg1")
    assert len(atts) == 1
    assert atts[0].filename == "doc.pdf"
    conn.close()


def test_embedding_exists(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message())
    assert not embedding_exists(conn, "msg1", None, "message", "test-model")
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id="msg1",
            attachment_id=None,
            chunk_type="message",
            chunk_text="test",
            embedding=b"\x00" * 12,
            model="test-model",
        ),
    )
    assert embedding_exists(conn, "msg1", None, "message", "test-model")
    conn.close()


def test_load_all_embeddings(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message())
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id="msg1",
            attachment_id=None,
            chunk_type="message",
            chunk_text="test",
            embedding=b"\x00" * 12,
            model="test-model",
        ),
    )
    ids, blobs = load_all_embeddings(conn, model="test-model")
    assert len(ids) == 1
    assert len(blobs) == 1
    conn.close()


def test_sync_state(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    assert get_sync_state(conn, "last_history_id") is None
    set_sync_state(conn, "last_history_id", "99999")
    assert get_sync_state(conn, "last_history_id") == "99999"
    set_sync_state(conn, "last_history_id", "100000")
    assert get_sync_state(conn, "last_history_id") == "100000"
    conn.close()


# ─── Drive stub round-trip ─────────────────────────────────────────────────
# upsert_drive_stub + fill_drive_attachment weren't previously covered by
# any test. The pair represents the full lifecycle of a Drive-linked doc
# row: stub insert (idempotent), then fill with fetched title/content.


def test_upsert_drive_stub_and_fill(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message())

    # First insert creates the stub row.
    inserted = upsert_drive_stub(
        conn,
        message_id="msg1",
        drive_id="drive_abc",
        mime_type="application/vnd.google-apps.document",
    )
    assert inserted == 1

    # Second insert is a no-op thanks to (message_id, filename) dedup.
    inserted_again = upsert_drive_stub(
        conn,
        message_id="msg1",
        drive_id="drive_abc",
        mime_type="application/vnd.google-apps.document",
    )
    assert inserted_again == 0

    atts = get_attachments_for_message(conn, "msg1")
    assert len(atts) == 1
    assert atts[0].filename == "Drive: [drive_abc]"
    stub_id = atts[0].id

    # Fill the stub with fetched content — renames filename to include title.
    fill_drive_attachment(
        conn,
        attachment_id=stub_id,
        title="My Doc",
        text="fetched content",
        drive_id="drive_abc",
    )
    conn.commit()

    atts = get_attachments_for_message(conn, "msg1")
    assert len(atts) == 1
    assert atts[0].filename == "Drive: My Doc [drive_abc]"
    assert atts[0].extracted_text == "fetched content"
    assert atts[0].size_bytes == len("fetched content")
    conn.close()
