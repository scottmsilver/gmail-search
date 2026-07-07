"""Tests for the LLM alias pipeline (aliases_llm.py) with a mocked backend.

The decisive behavior under test is the CAPS-DOMINANCE GATE: the 2026-07-07
eval showed a correct expansion for an ambiguous token ("net", "star") makes
search WORSE, so only caps-dominant tokens may reach the LLM at all.
"""

import json
import struct
from datetime import datetime

import gmail_search.aliases_llm as al
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_message


def _add_message(conn, i: int, subject: str, body: str) -> None:
    mid = f"al{i:05d}"
    upsert_message(
        conn,
        Message(
            id=mid,
            thread_id="t1",
            from_addr="a@b.com",
            to_addr="c@d.com",
            subject=subject,
            body_text=body,
            body_html="",
            date=datetime(2025, 1, 1),
            labels=[],
            history_id=1,
            raw_json="{}",
        ),
    )
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id=mid,
            attachment_id=None,
            chunk_type="message",
            chunk_text=body,
            embedding=struct.pack("4f", 0.1, 0.2, 0.3, 0.4),
            model="test-model",
        ),
    )


class _FakeBackend:
    """Answers WPC correctly; declares NET a non-abbreviation."""

    def __init__(self):
        self.asked: list[str] = []

    def chat(self, client, messages, *, max_tokens, json_format=False):
        token = messages[1]["content"].split('"')[1]
        self.asked.append(token)
        if token == "WPC":
            return json.dumps({"is_abbreviation": True, "expansions": ["west pacific colony"], "confidence": 0.9})
        return json.dumps({"is_abbreviation": False, "expansions": [], "confidence": 0.9})


def _build_corpus(db):
    conn = get_connection(db)
    # WPC: caps-dominant (always "WPC"), df >= MIN_DF. The body also spells
    # out the expansion so the corpus-grounding gate can verify it.
    for i in range(8):
        _add_message(conn, i, f"WPC update {i}", "The WPC (west pacific colony) board approved the budget.")
    # NET: caps df >= MIN_DF but the word "net" dominates -> gated out pre-LLM
    for i in range(8):
        _add_message(conn, 100 + i, f"NET alert {i}", "net metering and net income and net of fees " * 3)
    # RARE: caps-only but df < MIN_DF
    _add_message(conn, 200, "RARE thing", "RARE appears once.")
    conn.commit()
    conn.close()


def test_llm_alias_rebuild_gates_and_writes(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    init_db(db)
    _build_corpus(db)

    fake = _FakeBackend()
    monkeypatch.setattr("gmail_search.llm.openrouter.OpenRouterBackend", lambda: fake)
    n = al.rebuild_term_aliases_llm(db, data_dir=tmp_path)

    # NET never reaches the LLM (caps-dominance gate); RARE fails the df floor.
    assert "NET" not in fake.asked and "RARE" not in fake.asked
    assert "WPC" in fake.asked
    assert n == 1
    conn = get_connection(db)
    rows = conn.execute("SELECT term, expansions FROM term_aliases").fetchall()
    conn.close()
    table = {r["term"]: json.loads(r["expansions"]) for r in rows}
    assert table == {"wpc": ["west pacific colony"]}


def test_llm_alias_rebuild_replaces_previous_rows(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    init_db(db)
    _build_corpus(db)
    conn = get_connection(db)
    from gmail_search.auth.write_user import resolve_write_user_id

    uid = resolve_write_user_id(conn)
    conn.execute(
        "INSERT INTO term_aliases (term, expansions, similarity, user_id) VALUES (%s, %s, %s, %s)",
        ("stale", json.dumps(["old garbage"]), 0.5, uid),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("gmail_search.llm.openrouter.OpenRouterBackend", lambda: _FakeBackend())
    al.rebuild_term_aliases_llm(db, data_dir=tmp_path)

    conn = get_connection(db)
    terms = {r["term"] for r in conn.execute("SELECT term FROM term_aliases").fetchall()}
    conn.close()
    assert "stale" not in terms and "wpc" in terms


def test_llm_alias_rebuild_without_key_leaves_table_alone(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    init_db(db)
    _build_corpus(db)
    conn = get_connection(db)
    from gmail_search.auth.write_user import resolve_write_user_id

    uid = resolve_write_user_id(conn)
    conn.execute(
        "INSERT INTO term_aliases (term, expansions, similarity, user_id) VALUES (%s, %s, %s, %s)",
        ("keepme", json.dumps(["kept"]), 0.5, uid),
    )
    conn.commit()
    conn.close()

    def _raise():
        raise RuntimeError("OPENROUTER_KEY is not set; cannot use the OpenRouter backend")

    monkeypatch.setattr("gmail_search.llm.openrouter.OpenRouterBackend", _raise)
    assert al.rebuild_term_aliases_llm(db, data_dir=tmp_path) == -1
    conn = get_connection(db)
    terms = {r["term"] for r in conn.execute("SELECT term FROM term_aliases").fetchall()}
    conn.close()
    assert terms == {"keepme"}


def test_llm_alias_rebuild_removes_cooc_state(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    init_db(db)
    _build_corpus(db)
    conn = get_connection(db)
    from gmail_search.auth.write_user import resolve_write_user_id

    uid = resolve_write_user_id(conn)
    conn.close()
    cooc = tmp_path / "users" / uid / "aliases" / "cooc"
    cooc.mkdir(parents=True)
    (cooc / "big.run").write_bytes(b"x" * 128)

    monkeypatch.setattr("gmail_search.llm.openrouter.OpenRouterBackend", lambda: _FakeBackend())
    al.rebuild_term_aliases_llm(db, data_dir=tmp_path)
    assert not cooc.exists()


def test_sanitize_expansion_blocks_operator_injection():
    """Expansions come from an LLM that read attacker-controllable email
    content, and they're appended to queries BEFORE parse_query — so query
    operators, quotes, and control characters must never survive."""
    ok = al._sanitize_expansion
    assert ok("homeowners association") == "homeowners association"
    assert ok("AT&T wireless") == "at&t wireless"
    assert ok("d'angelo co.") == "d'angelo co."
    # operator smuggling / structure
    assert ok("from:evil@x.com") is None
    assert ok("subject:secret") is None
    assert ok('"quoted phrase"') is None
    assert ok("a\nb") is None
    # oversize
    assert ok("x" * 41) is None
    assert ok("one two three four five") is None


def test_grounding_drops_hallucinated_expansions(tmp_path, monkeypatch):
    """An expansion phrase that never occurs in the user's mail is a
    common-knowledge hallucination (WPC -> 'world peace council' when WPC
    is the user's property) and must not reach the alias table."""

    class _Hallucinator:
        def chat(self, client, messages, *, max_tokens, json_format=False):
            token = messages[1]["content"].split('"')[1]
            if token == "WPC":
                return json.dumps({
                    "is_abbreviation": True,
                    "expansions": ["world peace council", "west pacific colony"],
                    "confidence": 0.9,
                })
            return json.dumps({"is_abbreviation": False, "expansions": [], "confidence": 0.9})

    db = tmp_path / "t.db"
    init_db(db)
    _build_corpus(db)
    monkeypatch.setattr("gmail_search.llm.openrouter.OpenRouterBackend", lambda: _Hallucinator())
    al.rebuild_term_aliases_llm(db, data_dir=tmp_path)

    conn = get_connection(db)
    rows = {r["term"]: json.loads(r["expansions"]) for r in conn.execute("SELECT term, expansions FROM term_aliases").fetchall()}
    conn.close()
    # grounded expansion survives; the hallucination is gone
    assert rows.get("wpc") == ["west pacific colony"]
