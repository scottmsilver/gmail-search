"""The BM25 query builders treat every token as literal text.

Tantivy's parser gives meaning to `'` (phrase delimiter), to the upper-case
keywords `AND OR NOT IN TO`, and to a handful of punctuation. The sanitizer
strips the punctuation but deliberately keeps apostrophes and cannot know the
keyword list, so the builder quotes each token and the parser never sees bare
user text.
"""
import pytest

from gmail_search.store import queries


@pytest.mark.parametrize("token", ["PEET'S", "IN", "TO", "e-mail", "invoice"])
def test_disjunction_quotes_every_token(token):
    disjunction, phrase = queries._build_bm25_query([token], ("subject", "body_text"))
    assert disjunction == f'subject:"{token}" body_text:"{token}"'
    assert phrase is None


def test_phrase_pass_is_quoted_and_the_disjunction_keeps_quoting():
    disjunction, phrase = queries._build_bm25_query(["ALASKA", "IN"], ("subject",))
    assert disjunction == 'subject:"ALASKA" subject:"IN"'
    assert phrase == 'subject:"ALASKA IN"'


def test_sanitizer_never_emits_a_quote_or_backslash():
    tokens = queries._sanitize_fts_tokens('say "hi" \\ peet\'s ALASKA IN "flight')
    assert tokens
    assert not any('"' in t or "\\" in t for t in tokens)
