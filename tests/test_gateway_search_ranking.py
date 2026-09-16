"""Owner-revalidated hybrid candidates preserve the existing thread ranking."""
import importlib
import json
from datetime import datetime, timezone

import pytest

from gmail_search.gateway.search_queries import EmbeddingHit, MessageRow, ThreadRow

NOW = datetime.now(timezone.utc).isoformat()


def _embedding(identifier, message='m1', thread='t1', chunk='message', filename=None):
    return EmbeddingHit(id=identifier,message_id=message,attachment_id=None,chunk_type=chunk,
        chunk_text='matching text',thread_id=thread,subject='Draw request',from_addr='alice@test',
        date=NOW,att_filename=filename,model='test-model',chunk_bytes=13,chunk_complete=True)


def _message(message='m2',thread='t2'):
    return MessageRow(message_id=message,thread_id=thread,subject='Other subject',from_addr='bob@test',
        to_addr='me@test',date=NOW,body_text='keyword text',labels='[]',summary=None,body_bytes=12,body_complete=True,summary_bytes=None,summary_complete=True)


def _summary(thread='t1'):
    return ThreadRow(thread_id=thread,subject='Draw request',participants=json.dumps(['alice@test','me@test']),
        all_from_addrs=json.dumps(['alice@test','me@test']),all_labels=json.dumps(['IMPORTANT']),
        date_first=NOW,date_last=NOW,message_count=4)


def _rank(**kwargs):
    module=importlib.import_module('gmail_search.gateway.search_ranking')
    options=dict(query='draw request',temporal_boost=0.,vector_scores={1:.8,2:.4},
        embeddings=(_embedding(1),_embedding(2,'m2','t2')),lexical_scores={'m1':1.,'m2':0.},
        messages=(_message(),),summaries=(_summary(),),owner_emails=('me@test',),contact_frequency={},top_k=10)
    options.update(kwargs)
    return module.rank_candidates(**options)


def test_hybrid_merge_groups_chunks_and_keeps_best_message_match():
    result=_rank(vector_scores={1:.8,2:.4,3:.9},
        embeddings=(_embedding(1),_embedding(2,'m2','t2'),_embedding(3,chunk='attachment',filename='invoice.pdf')))
    first=result.threads[0]
    assert first.thread_id == 't1' and len(first.matches) == 1
    assert first.matches[0].match_type == 'attachment'
    assert first.matches[0].attachment_filename == 'invoice.pdf'
    assert first.similarity == .9 and first.message_count == 4 and first.user_replied
    assert 'missing_thread_summary' in result.reasons


def test_foreign_or_stale_index_score_cannot_influence_owner_normalization():
    baseline=_rank()
    extra=_rank(vector_scores={1:.8,2:.4,999:10000.})
    assert [t.thread_id for t in extra.threads] == [t.thread_id for t in baseline.threads]
    assert [t.score for t in extra.threads] == pytest.approx([t.score for t in baseline.threads])
    assert 'unavailable_embedding' in extra.reasons


def test_keyword_only_message_is_retained_and_missing_hydration_reported():
    result=_rank(vector_scores={},embeddings=(),lexical_scores={'m2':1.,'missing':.9})
    assert len(result.threads) == 1
    assert result.threads[0].matches[0].match_type == 'keyword'
    assert result.threads[0].similarity == 0
    assert 'unavailable_message' in result.reasons


def test_blend_retains_existing_weights_and_fresh_exact_subject_bonus():
    result=_rank(vector_scores={1:.8,2:.4},contact_frequency={'alice@test':.5})
    first=result.threads[0]
    # Top semantic=1, BM25=1, IMPORTANT=.35, replied=1, one of four messages
    # matches, size log(4)/log(50), contact=.5.
    #
    # Recency is computed rather than assumed to be 1. `NOW` is captured when
    # this module is imported and `_recency_score` decays exponentially from the
    # current time, so the gap between the two grows with however long the suite
    # takes to reach this test. At ~1e-5 tolerance that is minutes: it passed
    # per-file and failed in a 39-minute full run, which is the worst kind of
    # flake because the signal looks like a ranking regression.
    import math

    from gmail_search.search.ranking import _recency_score
    recency=_recency_score(NOW)
    expected=(.4+.15+.15*recency+.12*.35+.08+.06*.25
              +.04*math.log(4)/math.log(50)+.08*.5+.18*recency)
    assert first.score == pytest.approx(expected,abs=1e-5)
    # The decay must stay the only moving part; if the blend itself changed,
    # this bound would not save the test.
    assert .99 < recency <= 1


def test_malformed_private_summary_is_sanitized():
    row=_summary()
    from dataclasses import replace
    with pytest.raises(RuntimeError,match='Invalid search metadata') as error:
        _rank(summaries=(replace(row,participants='private broken JSON'),))
    assert 'private' not in str(error.value)


def test_semantic_singleton_keeps_legacy_zero_normalized_score():
    result=_rank(vector_scores={1:.8},embeddings=(_embedding(1),),lexical_scores={},messages=())
    assert result.threads[0].similarity == .8
    assert result.threads[0].score < .7


def test_recent_sort_and_temporal_intent_preserve_candidate_set():
    from dataclasses import replace
    older=replace(_summary('t2'),date_last='2000-01-01',subject='Old subject')
    result=_rank(summaries=(_summary(),older),sort='recent',temporal_boost=.35)
    assert [t.thread_id for t in result.threads] == ['t1','t2']
    assert not result.reasons


@pytest.mark.parametrize('field',['participants','all_from_addrs','all_labels'])
def test_summary_json_must_be_string_list(field):
    from dataclasses import replace
    with pytest.raises(RuntimeError,match='Invalid search metadata'):
        _rank(summaries=(replace(_summary(),**{field:'{"private":"value"}'}),))


@pytest.mark.parametrize('scores',[{'m1':float('nan')},{'m1':True},{'m1':float('inf')}])
def test_invalid_lexical_scores_fail_with_sanitized_error(scores):
    with pytest.raises(RuntimeError,match='Invalid search metadata'):
        _rank(lexical_scores=scores)
