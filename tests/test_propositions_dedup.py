"""Opt-in query-time near-duplicate collapse in find_facts (Feature 3)."""

import struct


class FakeEmb:
    model = "fake-embed"
    dimensions = 4

    def embed_query(self, q):
        return [1.0, 1.0, 0.0, 0.0]  # ~0.707 cosine to both groups below


def _blob(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def _seed(conn, uid, rows):
    from gmail_search import propositions as P

    P.ensure_table(conn)
    P.ensure_bm25_index(conn)
    for i, (text, vec) in enumerate(rows):
        conn.execute(
            "INSERT INTO propositions (user_id, message_id, thread_id, text, embedding, model, date) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (uid, f"m{i}", f"t{i}", text, _blob(vec), "m", "2020"),
        )
    conn.commit()


def _facts(out):
    return [r["fact"] for r in out]


def test_collapse_merges_near_dup_trio(db_backend, tmp_path):
    from gmail_search import propositions as P
    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.store.db import get_connection, init_db

    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    uid = resolve_write_user_id(conn)
    tesla = [1.0, 0.0, 0.0, 0.0]
    _seed(
        conn,
        uid,
        [
            ("Scott Silver owns a Tesla", tesla),
            ("Scott Silver has a Tesla", tesla),
            ("Scott Silver has a Tesla vehicle", tesla),
        ],
    )
    out = P.find_facts(conn, FakeEmb(), user_id=uid, query="tesla", hybrid=False, collapse_near_dups=True)
    assert len(out) == 1, _facts(out)
    assert out[0]["dup_count"] == 2
    assert "_pid" not in out[0]  # internal key stripped


def test_collapse_keeps_facts_with_conflicting_values(db_backend, tmp_path):
    from gmail_search import propositions as P
    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.store.db import get_connection, init_db

    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    uid = resolve_write_user_id(conn)
    same = [0.0, 1.0, 0.0, 0.0]  # identical embeddings (cosine 1.0)
    _seed(
        conn,
        uid,
        [
            ("vehicle registration expires in 2022", same),
            ("vehicle registration expires in 2024", same),
        ],
    )
    out = P.find_facts(conn, FakeEmb(), user_id=uid, query="registration", hybrid=False, collapse_near_dups=True)
    # Despite identical vectors, the differing year blocks the merge.
    assert len(out) == 2, _facts(out)


def test_collapse_off_reproduces_all_and_no_internal_keys(db_backend, tmp_path):
    from gmail_search import propositions as P
    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.store.db import get_connection, init_db

    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    uid = resolve_write_user_id(conn)
    tesla = [1.0, 0.0, 0.0, 0.0]
    _seed(
        conn,
        uid,
        [
            ("Scott Silver owns a Tesla", tesla),
            ("Scott Silver has a Tesla", tesla),
            ("Scott Silver has a Tesla vehicle", tesla),
        ],
    )
    out = P.find_facts(conn, FakeEmb(), user_id=uid, query="tesla", hybrid=False, collapse_near_dups=False)
    assert len(out) == 3
    for r in out:
        assert "dup_count" not in r and "_pid" not in r
