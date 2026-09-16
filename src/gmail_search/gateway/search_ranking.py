"""Hybrid thread ranking over rows already revalidated by the owner reader.

No database, filesystem, provider, or bootstrap identity access. Missing
hydration is explicit coverage metadata. Unhydrated vector IDs never influence
normalization or ordering. Reranking and the final off-topic filter follow this
step in the run service.
"""
from dataclasses import dataclass
import json
import math

from gmail_search.search import ranking as r


@dataclass(frozen=True)
class RankedCandidates:
    threads: tuple[r.ThreadResult, ...]
    reasons: tuple[str, ...]


def _strings(value):
    try:
        result = json.loads(value)
    except (ValueError, TypeError, RecursionError):
        raise RuntimeError('Invalid search metadata') from None
    if not isinstance(result, list) or len(result) > 10_000 or any(type(v) is not str for v in result):
        raise RuntimeError('Invalid search metadata')
    return result


def rank_candidates(*, query, temporal_boost, vector_scores, embeddings,
                    lexical_scores, messages, summaries, owner_emails,
                    contact_frequency, top_k, sort='relevance'):
    """Preserve the existing score blend and repeated-sender collapse.

    The bounded service owns input collection limits and owner binding. This
    internal function accepts no connections, index paths or model selectors.
    """
    if (type(top_k) is not int or not 1 <= top_k <= 100 or sort not in ('relevance','recent')
            or type(temporal_boost) not in (int,float) or not math.isfinite(temporal_boost)
            or not 0 <= temporal_boost <= .35):
        raise ValueError('Invalid search ranking options')
    if any(type(score) not in (float,int) or not math.isfinite(score)
           for scores in (vector_scores,lexical_scores,contact_frequency) for score in scores.values()):
        raise RuntimeError('Invalid search metadata')
    reasons = set()
    rows = {row.id: row for row in embeddings}
    message_map = {row.message_id: row for row in messages}
    summary_map = {row.thread_id: row for row in summaries}
    valid_scores = {identifier: score for identifier, score in vector_scores.items() if identifier in rows}
    if len(valid_scores) != len(vector_scores):
        reasons.add('unavailable_embedding')
    max_sim = max(valid_scores.values(), default=1.)
    min_sim = min(valid_scores.values(), default=0.)
    sim_range = max_sim - min_sim if max_sim != min_sim else 1.
    threads = {}
    for identifier, score in valid_scores.items():
        row = rows[identifier]
        normalized = (score - min_sim) / sim_range
        thread = threads.setdefault(row.thread_id, dict(best_sim=normalized,raw_sim=score,
                                                       subject=row.subject,matches={}))
        if normalized > thread['best_sim']:
            thread['best_sim'], thread['raw_sim'] = normalized, score
        previous = thread['matches'].get(row.message_id)
        if previous is None or score > previous.score:
            thread['matches'][row.message_id] = r.ThreadMatch(
                row.message_id,score,row.from_addr,row.date,(row.chunk_text or '')[:200],
                row.chunk_type,row.att_filename)
    matched = {mid for thread in threads.values() for mid in thread['matches']}
    for mid in lexical_scores:
        if mid in matched:
            continue
        row = message_map.get(mid)
        if row is None:
            reasons.add('unavailable_message')
            continue
        thread = threads.setdefault(row.thread_id, dict(best_sim=0.,raw_sim=0.,subject=row.subject,matches={}))
        thread['matches'][mid] = r.ThreadMatch(mid,0.,row.from_addr,row.date,
                                             (row.body_text or '')[:200],'keyword')
    thread_bm25 = {tid:max((lexical_scores.get(mid,0.) for mid in thread['matches']),default=0.)
                   for tid,thread in threads.items()}
    max_bm25 = max(thread_bm25.values(),default=0.)
    output = []
    for tid, thread in threads.items():
        matches = sorted(thread['matches'].values(),key=lambda match:match.score,reverse=True)
        summary = summary_map.get(tid)
        if summary is not None:
            participants = _strings(summary.participants)
            from_addrs = _strings(summary.all_from_addrs)
            labels = _strings(summary.all_labels)
            count = summary.message_count
            if type(count) is not int or count < 1:
                raise RuntimeError('Invalid search metadata')
            date_first,date_last,subject = summary.date_first,summary.date_last,summary.subject
        else:
            reasons.add('missing_thread_summary')
            participants = sorted({m.from_addr for m in matches})
            from_addrs = [m.from_addr for m in matches]
            labels = []
            count = len(matches)
            date_first,date_last = matches[-1].date,matches[0].date
            subject = thread['subject']
        replied = any(email in address.lower() for address in from_addrs for email in owner_emails)
        similarity,bm25 = thread['best_sim'],thread_bm25[tid]
        recency = r._recency_score(date_last)
        blended = ((r.W_SIMILARITY-temporal_boost)*similarity + r.W_BM25*bm25
                   + (r.W_RECENCY+temporal_boost)*recency + r.W_LABELS*r._label_score([labels])
                   + r.W_REPLIED*float(replied) + r.W_MATCH_DENSITY*r._match_density_score(len(matches),count)
                   + r.W_THREAD_SIZE*r._thread_size_score(count)
                   + .08*r._contact_frequency_score(from_addrs,contact_frequency))
        strength = max(similarity,bm25/max_bm25 if max_bm25>0 else 0.,r._exact_subject_phrase(query,subject))
        blended += r.W_FRESH_MATCH*strength*recency
        if not math.isfinite(blended):
            raise RuntimeError('Invalid search metadata')
        output.append(r.ThreadResult(tid,blended,thread['raw_sim'],subject,participants,count,
                                     date_first,date_last,replied,matches))
    output.sort(key=(lambda thread:thread.date_last) if sort=='recent' else (lambda thread:thread.score),reverse=True)
    return RankedCandidates(tuple(r.collapse_repeat_senders(output,top_k*2)),tuple(sorted(reasons)))
