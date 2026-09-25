"""Pure ranking is reusable without opening legacy DB/provider paths."""
import importlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest


def test_ranking_import_does_not_load_database_provider_or_native_index():
    result = subprocess.run([sys.executable, '-c',
        "import json,sys; import gmail_search.search.ranking; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    modules = json.loads(result.stdout)
    assert not any(name.startswith(('gmail_search.store', 'gmail_search.embed',
                                   'gmail_search.index', 'numpy', 'google.genai')) for name in modules)


def test_ranking_signals_keep_existing_scale():
    r = importlib.import_module('gmail_search.search.ranking')
    assert r._label_score([['INBOX', 'STARRED'], ['STARRED']]) == pytest.approx(.4)
    assert r._label_score([['CATEGORY_PROMOTIONS']]) == 0
    assert r._match_density_score(2, 4) == .5
    assert r._match_density_score(4, 2) == 1
    assert r._thread_size_score(1) == 0
    assert r._thread_size_score(50) == 1
    assert r._exact_subject_phrase('draw request', 'May Draw Request') == 1
    assert r._exact_subject_phrase('draw request', 'Draw a new request') == 0
    assert r._recency_score('invalid') == 0
    assert r._recency_score((datetime.now(timezone.utc) - timedelta(days=60)).isoformat()) == pytest.approx(.5, abs=.001)
    assert r._contact_frequency_score(['Alice <a@example.test>'], {'a@example.test': .7}) == .7
    assert r._contact_frequency_score(['ALICE@Example.test, "Bob" <b@example.test>'],
                                      {'a@example.test': .7, 'b@example.test': .9}) == .9
    # Exact address match, not substring: bob@ is not jimbob@.
    assert r._contact_frequency_score(['jimbob@example.test'], {'bob@example.test': .9}) == 0.0


def test_legacy_engine_reexports_same_result_types_and_helpers():
    r = importlib.import_module('gmail_search.search.ranking')
    engine = importlib.import_module('gmail_search.search.engine')
    for name in ('SearchResult', 'ThreadMatch', 'ThreadResult', '_recency_score', '_label_score'):
        assert getattr(engine, name) is getattr(r, name)


def _thread(r, identifier, score, subject='Digest 1', sender='news@example.test', count=1):
    return r.ThreadResult(identifier, score, score, subject, [sender], count, '', '', False)


def test_repeat_sender_collapse_preserves_distinct_subjects_and_conversations():
    r = importlib.import_module('gmail_search.search.ranking')
    items = [_thread(r, str(i), 1 - i / 10, f'Digest {i}') for i in range(3)]
    items += [_thread(r, 'distinct', .6, 'Personal letter'),
              _thread(r, 'conversation', .5, 'Digest 7', count=2)]
    assert [t.thread_id for t in r.collapse_repeat_senders(items, 10)] == ['0', 'distinct', 'conversation']
    assert r.collapse_repeat_senders(items[:2], 10) == items[:2]
    assert r.normalize_subject('Re: Digest 2030') == 'digest #'


def test_offtopic_retains_at_least_three_and_preserves_threshold_boundary():
    r = importlib.import_module('gmail_search.search.ranking')
    items = [_thread(r, str(i), score) for i, score in enumerate([1, .8, .4, .399])]
    assert r.filter_offtopic(items) == items[:3]
    items[1].score = .2
    items[2].score = .1
    assert r.filter_offtopic(items) == items[:3]
    assert r.filter_offtopic([]) == []
    assert r.filter_offtopic([_thread(r, 'zero', 0)])
