"""Query-normalization for the BM25 path of find_facts.

Regression for the "what cars do I own" failure: interrogative/possession
stopwords used to dominate the lexical match (and return zero car facts) and
plurals didn't match singular fact text.
"""

from gmail_search.propositions import _query_terms, _singularize


def test_query_terms_strips_interrogatives_and_possessives():
    assert _query_terms("what cars do I own") == ["cars"]
    assert _query_terms("all my license plates") == ["license", "plates"]
    assert _query_terms("who are my doctors") == ["doctors"]


def test_query_terms_keeps_bare_alphanumeric_identifiers():
    # Plates/VINs must survive — they are the hardest recall case.
    assert _query_terms("7ABC123") == ["7abc123"]
    assert _query_terms("VIN 5YJ3E1EA7KF") == ["vin", "5yj3e1ea7kf"]


def test_singularize_common_plurals():
    assert _singularize("cars") == "car"
    assert _singularize("plates") == "plate"
    assert _singularize("batteries") == "battery"
    assert _singularize("boxes") == "box"
    assert _singularize("glasses") == "glass"


def test_singularize_returns_none_when_no_distinct_singular():
    for tok in ("car", "vin", "address", "license", "7abc123", "tesla", "gas"):
        assert _singularize(tok) is None, tok


def test_bm25_query_construction(monkeypatch):
    # Capture the tsquery string _bm25_ids builds without touching a DB.
    import gmail_search.propositions as P

    captured = {}

    class FakeConn:
        def execute(self, sql, params):
            captured["pq"] = params[1]

            class R:
                def fetchall(self_inner):
                    return []

            return R()

    P._bm25_ids(FakeConn(), "u1", "what cars do I own", 10)
    assert captured["pq"] == "(text:cars OR text:car)"

    P._bm25_ids(FakeConn(), "u1", "7ABC123", 10)
    assert captured["pq"] == "text:7abc123"

    P._bm25_ids(FakeConn(), "u1", "all my license plates", 10)
    assert captured["pq"] == "text:license OR (text:plates OR text:plate)"
