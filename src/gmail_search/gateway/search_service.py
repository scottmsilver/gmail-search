"""Run-bound hybrid search orchestration; no public mounting or provider fallback.

Provider adapters are trusted, budgeted dependencies. They must drain network
work and settle reservations on cancellation. This component does not qualify
those adapters or replace their accounting with guest-supplied usage.
"""
import asyncio
from dataclasses import asdict, dataclass, field
import json
import math
import re
from types import MappingProxyType

from gmail_search.search.parser import parse_query
from gmail_search.search.ranking import filter_offtopic

from .registry import AccessDenied
from .search_index import PendingIndex
from .search_queries import Selection, StructuredFilters, _text
from .search_ranking import rank_candidates
from .search_vectors import exact_candidates


@dataclass(frozen=True)
class OwnerSearchContext:
    owner_id: str
    emails: tuple[str, ...]
    corrector: object = field(default=None,repr=False,compare=False)

    def __post_init__(self):
        _text(self.owner_id)
        if (not isinstance(self.emails,tuple) or not 1 <= len(self.emails) <= 32
                or any(type(email) is not str or not 3 <= len(email) <= 320 or '@' not in email
                       or any(c.isspace() for c in email) or '\x00' in email for email in self.emails)
                or (self.corrector is not None and not callable(self.corrector))):
            raise ValueError('Invalid trusted owner search context')
        object.__setattr__(self,'emails',tuple(dict.fromkeys(email.lower() for email in self.emails)))


async def _drain(task):
    interrupted=False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted=True
        except BaseException:
            break
    if interrupted:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def _thread(function,*args,**kwargs):
    # Threads cannot be killed; cancellation acknowledges their exit first.
    task=asyncio.create_task(asyncio.to_thread(function,*args,**kwargs))
    return await _drain(task)


def _tokens(text):
    tokens=re.findall(r"[\w]+(?:[-'][\w]+)*",text)
    if len(tokens)>128 or any(len(token)>128 for token in tokens):
        raise ValueError('Search query exceeds lexical limits')
    return tuple(token for token in tokens if token.upper() not in {'AND','OR','NOT','NEAR'})


def _remember(reasons,selection,phase):
    if not selection.complete:
        reasons.add(phase+'_'+(selection.reason or 'incomplete'))


class RunSearchService:
    def __init__(self,capabilities,reader,indexes,embedder,*,owners,reranker):
        if len(owners)>4096 or any(type(value) is not OwnerSearchContext or key!=value.owner_id
                                  for key,value in owners.items()):
            raise ValueError('Invalid trusted owner registry')
        correctors=[id(value.corrector) for value in owners.values() if value.corrector is not None]
        if len(correctors)!=len(set(correctors)):
            raise ValueError('Spell resources cannot be shared across owners')
        self.capabilities=capabilities
        self.reader=reader
        self.indexes=indexes
        self.embedder=embedder
        self.owners=MappingProxyType(dict(owners))
        self.reranker=reranker

    async def authorize(self,token):
        return await _thread(self.capabilities.authorize,token,audience='retrieval',operation='search')

    async def search(self,token,*,query,top_k=10,date_from=None,date_to=None,detail='snippet',max_matches=3):
        lease=await self.authorize(token)
        _text(query,1000)
        if (not query.strip() or type(top_k) is not int or not 1<=top_k<=100
                or detail not in ('refs','snippet','summary','full')
                or type(max_matches) is not int or not 0<=max_matches<=100):
            raise ValueError('Invalid search options')
        StructuredFilters(date_from=date_from,date_to=date_to)
        deadline=asyncio.get_running_loop().time()+30
        async def check():
            current=await self.authorize(token)
            if current.run_id!=lease.run_id or current.owner_id!=lease.owner_id:
                raise AccessDenied()
            if asyncio.get_running_loop().time()>=deadline:
                raise TimeoutError('Search deadline exceeded')
            return True
        async def watch():
            while True:
                await asyncio.sleep(.1)
                await check()
        async def execute():
            async with asyncio.timeout_at(deadline):
                return await self._execute(lease,query,top_k,date_from,date_to,detail,max_matches,deadline,check)
        work=asyncio.create_task(execute())
        watcher=asyncio.create_task(watch())
        try:
            done,_=await asyncio.wait((work,watcher),return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                await watcher
            result=await work
        finally:
            work.cancel()
            watcher.cancel()
            await _drain(asyncio.ensure_future(asyncio.gather(work,watcher,return_exceptions=True)))
        # Watcher authorization threads can finish after the search itself.
        # Drain them before the final fresh authorization/deadline check; no
        # further suspension may follow that check before publication.
        await check()
        return result

    async def _context(self,owner,deadline,check,reasons):
        aliases={}
        contacts={}
        async with self.reader.session(owner,deadline=deadline,check_active=check) as db:
            for name,target,key,value in (('owner_aliases',aliases,'term','expansions'),
                                           ('owner_contacts',contacts,'email','score')):
                after=None
                for _ in range(10):
                    page=await getattr(db,name)(after=after,limit=1000)
                    for row in page.rows:
                        identifier=getattr(row,key)
                        item=getattr(row,value)
                        if name=='owner_aliases':
                            try:
                                item=json.loads(item)
                            except (TypeError,ValueError,RecursionError):
                                raise RuntimeError('Invalid search context') from None
                            if not isinstance(item,list) or len(item)>100 or any(type(v) is not str or len(v)>1000 for v in item):
                                raise RuntimeError('Invalid search context')
                        elif type(item) not in (float,int) or not math.isfinite(item) or not 0<=item<=1:
                            raise RuntimeError('Invalid search context')
                        target[identifier.lower()]=item
                    if page.complete:
                        break
                    if page.next_cursor is None:
                        _remember(reasons,page,name)
                        break
                    if type(page.next_cursor) is not str or (after is not None and page.next_cursor<=after):
                        raise RuntimeError('Invalid search context')
                    after=page.next_cursor
                else:
                    reasons.add(name+'_limit')
        return aliases,contacts

    async def _lexical(self,db,text,candidates,limit,reasons):
        tokens=_tokens(text)
        allowed=None if candidates is None else frozenset(candidates)
        scores={}
        for phrase in ((True,False) if len(tokens)>1 else (False,)):
            for operation in (db.lexical_messages,db.lexical_attachments):
                page=await operation(tokens,phrase=phrase,candidate_ids=candidates,limit=limit)
                _remember(reasons,page,'lexical')
                for row in page.rows:
                    if type(row.score) not in (float,int) or not math.isfinite(row.score) or row.score<0:
                        raise RuntimeError('Invalid search score')
                    if allowed is not None and row.message_id not in allowed:
                        raise RuntimeError('Invalid search candidate')
                    score=row.score*(1.5 if phrase else 1.)
                    scores[row.message_id]=max(scores.get(row.message_id,0.),score)
        if not scores:
            return scores
        lo,hi=min(scores.values()),max(scores.values())
        return {mid:((score-lo)/(hi-lo) if hi!=lo else 1.) for mid,score in scores.items()}

    async def _hydrate(self,operation,ids,reasons,phase,**kwargs):
        # Lexical phrase/disjunction/original passes can union beyond one fixed
        # query's input cap. Page that union rather than silently dropping IDs.
        if len(ids)>20_000:
            raise RuntimeError('Search candidate union exceeds limit')
        rows=[]
        complete=True
        for offset in range(0,len(ids),1000):
            page=await operation(ids[offset:offset+1000],**kwargs)
            _remember(reasons,page,phase)
            complete=complete and page.complete
            rows.extend(page.rows)
            if not page.complete and page.reason=='session_budget':
                break
        return Selection(tuple(rows),complete,None if complete else 'incomplete')

    async def _execute(self,lease,query,top_k,date_from,date_to,detail,max_matches,deadline,check):
        owner=self.owners.get(lease.owner_id)
        if owner is None:
            raise RuntimeError('Search profile is unavailable')
        reasons=set()
        try:
            # Complete the fixed reader's schema/credential qualification before
            # native index loading. This snapshot closes before provider work.
            aliases,contacts=await self._context(lease.owner_id,deadline,check,reasons)
            async with self.indexes.acquire(lease.owner_id) as index:
                profile=self.reader.profile
                if (index.binding.owner_id!=lease.owner_id or index.binding.model!=profile.embedding_model
                        or index.binding.dimensions!=profile.dimensions or self.embedder.model!=profile.embedding_model
                        or self.embedder.dimensions!=profile.dimensions):
                    raise RuntimeError('Search profile is unavailable')
                cleaned=query if owner.corrector is None else await _thread(owner.corrector,query)
                _text(cleaned,1000)
                expanded=' '.join(part for word in cleaned.split() for part in (word,*aliases.get(word.lower(),[])[:2]))
                _text(expanded,8192)
                try:
                    parsed=parse_query(expanded)
                    original=parse_query(query)
                except (ValueError,OverflowError):
                    raise ValueError('Invalid search query') from None
                filters=StructuredFilters(parsed.from_filter,parsed.to_filter,parsed.subject_filter,parsed.has_attachment,
                                          date_from or parsed.date_from,date_to or parsed.date_to)
                vector=await self.embedder.embed(lease,parsed.text or expanded,deadline=deadline,check_active=check)
                await check()
                async with self.reader.session(lease.owner_id,deadline=deadline,check_active=check) as db:
                    candidates=None
                    structured=any(value is not None for value in asdict(filters).values())
                    if structured:
                        selected=await db.structured_candidates(filters)
                        _remember(reasons,selected,'structured')
                        candidates=tuple(row.message_id for row in selected.rows)
                    has_date=bool(filters.date_from or filters.date_to)
                    fetch_k=max(top_k*10,2000) if has_date else top_k*10
                    semantic='approximate'
                    if candidates is not None and len(candidates)<=20_000:
                        exact=await exact_candidates(db,candidates,vector,dimensions=profile.dimensions,top_k=fetch_k,
                                                     deadline=deadline,check_active=check)
                        ids,scores=exact.ids,exact.scores
                        reasons.update(exact.reasons)
                        semantic='exact_restricted'
                    else:
                        ids,scores=await index.search(vector,top_k=10_000 if candidates is not None else fetch_k,
                                                      absolute_deadline=deadline)
                        reasons.add('approximate_vector_search')
                    hydrated=await db.hydrate_embeddings(tuple(ids))
                    _remember(reasons,hydrated,'embedding_hydration')
                    allowed=None if candidates is None else frozenset(candidates)
                    rows=tuple(row for row in hydrated.rows if allowed is None or row.message_id in allowed)
                    vector_scores=dict(zip(ids,scores,strict=True))
                    if candidates is not None:
                        # Revalidate the overfetch first, then preserve only the
                        # first fetch_k permitted candidates, as the legacy path.
                        known={row.id for row in hydrated.rows}
                        if any(identifier not in known for identifier in ids):
                            reasons.add('unavailable_embedding')
                        permitted={row.id for row in rows}
                        selected_ids=[identifier for identifier in ids if identifier in permitted][:fetch_k]
                        vector_scores={identifier:vector_scores[identifier] for identifier in selected_ids}
                        rows=tuple(row for row in rows if row.id in vector_scores)
                    lexical=await self._lexical(db,parsed.text or expanded,candidates,2000 if has_date else 200,reasons)
                    if expanded.lower()!=query.lower():
                        other=await self._lexical(db,original.text or query,candidates,2000 if has_date else 200,reasons)
                        for mid,score in other.items():lexical[mid]=max(lexical.get(mid,0.),score)
                    messages=await self._hydrate(db.hydrate_messages,tuple(lexical),reasons,'message_hydration')
                    _remember(reasons,messages,'message_hydration')
                    thread_ids=tuple(dict.fromkeys([row.thread_id for row in rows]+[row.thread_id for row in messages.rows]))
                    summaries=await self._hydrate(db.hydrate_threads,thread_ids,reasons,'thread_hydration')
                    _remember(reasons,summaries,'thread_hydration')
                    ranked=await _thread(rank_candidates,query=parsed.text or expanded,temporal_boost=parsed.temporal_boost,
                        vector_scores=vector_scores,embeddings=rows,lexical_scores=lexical,messages=messages.rows,
                        summaries=summaries.rows,owner_emails=owner.emails,contact_frequency=contacts,top_k=top_k)
                    reasons.update(ranked.reasons)
                    threads=list(ranked.threads)
                    all_ids=tuple(dict.fromkeys(match.message_id for thread in threads for match in thread.matches))
                    topics=await self._hydrate(db.message_topics,all_ids,reasons,'topic_hydration')
                    _remember(reasons,topics,'topic_hydration')
                    facets=await db.topic_facets(all_ids)
                    _remember(reasons,facets,'topic_facets')
                    kept_ids=tuple(dict.fromkeys(match.message_id for thread in threads
                                                 for match in (thread.matches[:max_matches] if max_matches else thread.matches)))
                    details=await self._hydrate(db.hydrate_messages,kept_ids,reasons,'detail_hydration',
                                                body_chars=20_000 if detail=='full' else 200)
                    _remember(reasons,details,'detail_hydration')
                reranking='disabled' if self.reranker is None else 'not_needed'
                if self.reranker is not None and len(threads)>3:
                    top_scores=[thread.score for thread in threads[:5]]
                    if max(top_scores)-min(top_scores)<.05:
                        candidates_to_rank=tuple(threads[:30])
                        order=await self.reranker.rerank(lease,parsed.text or query,candidates_to_rank,
                                                        deadline=deadline,check_active=check)
                        expected={thread.thread_id for thread in candidates_to_rank}
                        if (not isinstance(order,(tuple,list)) or len(order)!=len(expected)
                                or any(type(tid) is not str for tid in order) or set(order)!=expected):
                            raise RuntimeError('Invalid search reranking response')
                        by_id={thread.thread_id:thread for thread in candidates_to_rank}
                        threads=[by_id[tid] for tid in order]+threads[30:]
                        reranking='applied'
                threads=filter_offtopic(threads)[:top_k]
                await check()
                return self._format(threads,details,topics,facets,detail,max_matches,reasons,semantic,reranking,owner.corrector is not None)
        except PendingIndex:
            return dict(results=[],facets=[],pending_index=True,coverage=dict(complete=False,reasons=['pending_index']))

    @staticmethod
    def _format(threads,details,topics,facets,detail,max_matches,reasons,semantic,reranking,spellcheck):
        by_id={row.message_id:row for row in details.rows}
        topic_map={}
        for row in topics.rows:
            topic_map.setdefault(row.message_id,set()).add(row.topic_id)
        leaf_labels={row.topic_id:row.label for row in facets.rows}
        facet_threads={}
        formatted=[]
        size=0
        for thread in threads:
            topic_ids=sorted({topic for match in thread.matches for topic in topic_map.get(match.message_id,())})
            if detail=='refs':
                result=dict(thread_id=thread.thread_id,cite_ref=thread.thread_id,subject=thread.subject,
                            date_last=thread.date_last,score=thread.score,
                            **{'from':thread.matches[0].from_addr if thread.matches else ''})
            else:
                result=asdict(thread)
                result['cite_ref']=thread.thread_id
                result['topic_ids']=topic_ids
                result['message_count_complete']='missing_thread_summary' not in reasons
                if max_matches and len(result['matches'])>max_matches:
                    result['matches_truncated']=len(result['matches'])-max_matches
                    result['matches']=result['matches'][:max_matches]
                for match in result['matches']:
                    match['cite_ref']=thread.thread_id
                    row=by_id.get(match['message_id'])
                    if detail in ('summary','full') and row is None:
                        reasons.add('unavailable_detail')
                    elif detail=='full':
                        match.update(body=row.body_text,body_format='text',body_complete=row.body_complete,
                                     body_bytes=row.body_bytes)
                        if not row.body_complete:reasons.add('body_truncated')
                    elif detail=='summary':
                        match.update(summary=row.summary or '',summary_complete=row.summary_complete,
                                     summary_model=row.summary_model,summary_created_at=row.summary_created_at)
                        if not row.summary_complete:reasons.add('summary_truncated')
            encoded=json.dumps(result,ensure_ascii=False,allow_nan=False).encode('utf-8')
            if size+len(encoded)>4*1024*1024-8192:
                reasons.add('response_bytes')
                break
            formatted.append(result)
            for topic in topic_ids:
                if topic in leaf_labels:
                    facet_threads.setdefault(topic,set()).add(thread.thread_id)
            size+=len(encoded)
        facet_output=[]
        for topic,tids in sorted(facet_threads.items(),key=lambda item:(-len(item[1]),item[0])):
            item=dict(topic_id=topic,label=leaf_labels[topic],count=len(tids))
            item_size=len(json.dumps(item,ensure_ascii=False,allow_nan=False).encode('utf-8'))
            if size+item_size>4*1024*1024-8192:
                reasons.add('facet_response_bytes')
                break
            facet_output.append(item)
            size+=item_size
        return dict(results=formatted,facets=facet_output,coverage=dict(complete=False,reasons=sorted(reasons),semantic=semantic,
                                                   reranking=reranking,spellcheck=spellcheck))
