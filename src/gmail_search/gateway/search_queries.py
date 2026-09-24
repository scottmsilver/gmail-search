"""Closed internal search operations. No guest SQL, owner, model or path selector."""
from dataclasses import dataclass
from datetime import date
import re
from typing import Generic, TypeVar

from psycopg import sql

T = TypeVar('T')


@dataclass(frozen=True)
class Selection(Generic[T]):
    rows: tuple[T, ...]
    complete: bool
    reason: str | None = None
    next_cursor: int | str | None = None


@dataclass(frozen=True)
class StructuredFilters:
    from_filter: str | None = None
    to_filter: str | None = None
    subject_filter: str | None = None
    has_attachment: bool | None = None
    date_from: str | None = None
    date_to: str | None = None

    def __post_init__(self):
        for value in (self.from_filter,self.to_filter,self.subject_filter):
            if value is not None:
                _text(value,1000)
        if self.has_attachment is not None and type(self.has_attachment) is not bool:
            raise ValueError('Invalid attachment filter')
        for value in (self.date_from,self.date_to):
            if value is not None:
                if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
                    raise ValueError('Invalid search date')
                date.fromisoformat(value)
        if self.date_from and self.date_to and self.date_from>self.date_to:
            raise ValueError('Invalid search date range')


@dataclass(frozen=True)
class MessageCandidate:
    message_id: str


@dataclass(frozen=True)
class LexicalHit:
    id: int
    message_id: str
    score: float

    def __post_init__(self):
        if type(self.id) is not int:
            raise ValueError('Invalid numeric lexical key')


@dataclass(frozen=True)
class TextMessageLexicalHit:
    id: str
    message_id: str
    score: float

    def __post_init__(self):
        if type(self.id) is not str or not self.id or self.id != self.message_id:
            raise ValueError('Invalid TEXT message lexical key')


@dataclass(frozen=True)
class EmbeddingHit:
    id: int
    message_id: str
    attachment_id: int | None
    chunk_type: str
    chunk_text: str | None
    chunk_bytes: int | None
    chunk_complete: bool
    thread_id: str
    subject: str
    from_addr: str
    date: str
    att_filename: str | None
    model: str


@dataclass(frozen=True)
class MessageRow:
    message_id: str
    thread_id: str
    subject: str
    from_addr: str
    to_addr: str
    date: str
    body_text: str
    body_bytes: int
    body_complete: bool
    labels: str
    summary: str | None
    summary_bytes: int | None
    summary_complete: bool
    summary_model: str | None = None
    summary_created_at: str | None = None


@dataclass(frozen=True)
class ThreadRow:
    thread_id: str
    subject: str
    participants: str
    all_from_addrs: str
    all_labels: str
    date_first: str
    date_last: str
    message_count: int


@dataclass(frozen=True)
class FactVectorRow:
    id: int
    message_id: str
    thread_id: str | None
    text: str
    embedding: bytes | None
    model: str
    vector_status: str


@dataclass(frozen=True)
class EmbeddingVectorRow:
    id: int
    message_id: str
    embedding: bytes | None
    model: str
    vector_status: str


@dataclass(frozen=True)
class AliasRow:
    term: str
    expansions: str


@dataclass(frozen=True)
class ContactRow:
    email: str
    score: float


@dataclass(frozen=True)
class TopicRow:
    message_id: str
    topic_id: str
    label: str


@dataclass(frozen=True)
class TopicFacet:
    topic_id: str
    label: str
    count: int


def _text(value, maximum=2048):
    if not isinstance(value,str) or not value or len(value)>maximum or '\x00' in value:
        raise ValueError('Invalid search text')
    try:
        value.encode('utf-8')
    except UnicodeError:
        raise ValueError('Invalid search text') from None
    return value


def _limit(value, maximum):
    if type(value) is not int or not 1<=value<=maximum:
        raise ValueError('Invalid search limit')
    return value


def _ids(values, *, numeric=False, maximum=100_000):
    if not isinstance(values,(tuple,list)) or len(values)>maximum:
        raise ValueError('Invalid search ID collection')
    for value in values:
        if numeric:
            if type(value) is not int or not -(2**63)<=value<2**63:
                raise ValueError('Invalid search ID')
        else:
            _text(value)
    if sum(len(str(value).encode()) for value in values)>4_000_000:
        raise ValueError('Search ID input exceeds limit')
    return tuple(dict.fromkeys(values))


def lexical_query(branch, tokens, *, phrase=False):
    fields={'messages':('subject','from_addr','to_addr','body_text'),
            'attachments':('filename','extracted_text'),'facts':('text',)}
    if branch not in fields or type(phrase) is not bool or not isinstance(tokens,(tuple,list)) or len(tokens)>128:
        raise ValueError('Invalid lexical request')
    for token in tokens:
        if not isinstance(token,str) or not re.fullmatch(r"[\w]+(?:[-'][\w]+)*",token) or len(token)>128:
            raise ValueError('Invalid lexical token')
        _text(token,128)
    if sum(len(token.encode()) for token in tokens)>8192:
        raise ValueError('Lexical input exceeds limit')
    if phrase:
        return ' '.join(field+':"'+' '.join(tokens)+'"' for field in fields[branch]) if tokens else ''
    return ' '.join(field+':'+token for field in fields[branch] for token in tokens)


class BoundSearchQueries:
    """One bounded snapshot; constructed only by SearchReader.session.

    Selection.complete describes this fixed selection, not whole-mailbox
    semantic recall. Byte/row limits never turn an error into an empty success.
    """
    def __init__(self, session):
        self._session=session

    async def structured_candidates(self, filters: StructuredFilters, *, limit=100_000):
        if not isinstance(filters,StructuredFilters):
            raise ValueError('Invalid structured filters')
        _limit(limit,100_000)
        where=['m.user_id=%s'];params=[self._session.owner_id]
        for field,value in (('from_addr',filters.from_filter),('to_addr',filters.to_filter),('subject',filters.subject_filter)):
            if value is not None:
                # Preserve existing ILIKE wildcard semantics; values remain bound.
                where.append('m.'+field+' ILIKE %s');params.append('%'+value+'%')
        if filters.date_from:
            where.append('m.date>=%s');params.append(filters.date_from+'T00:00:00')
        if filters.date_to:
            where.append('m.date<=%s');params.append(filters.date_to+'T23:59:59')
        if filters.has_attachment is not None:
            where.append(('' if filters.has_attachment else 'NOT ')+'EXISTS(SELECT 1 FROM public.attachments a WHERE a.user_id=m.user_id AND a.message_id=m.id)')
        return await self._session.read_message_ids('SELECT m.user_id AS owner_id,m.id AS message_id FROM public.messages m WHERE '+' AND '.join(where)+' ORDER BY m.id',params,limit)

    async def _lexical(self,branch,tokens,*,phrase=False,candidate_ids=None,limit=200):
        _limit(limit,10_000)
        query=lexical_query(branch,tokens,phrase=phrase)
        candidates=None if candidate_ids is None else _ids(candidate_ids)
        if not query or candidates==():
            return Selection((),True)
        profile=self._session.profile.schema_profile
        table,key,mid={'messages':('messages',profile.message_key,'id'),'attachments':('attachments','id','message_id'),'facts':('propositions','id','message_id')}[branch]
        # Native partition generic plans require a constant owner restriction.
        # This literal comes only from the immutable credential, never a method argument.
        owner=sql.Literal(self._session.owner_id).as_string(self._session.conn)
        statement=f'SELECT user_id AS owner_id,{key} AS id,{mid} AS message_id,paradedb.score({key}) AS score FROM public.{table} WHERE user_id={owner} AND {key} OPERATOR(pg_catalog.@@@) %s'
        params=[query]
        if candidates is not None:
            statement+=f' AND {mid}=ANY(%s::text[])';params.append(list(candidates))
        row_type=TextMessageLexicalHit if branch=='messages' and key=='id' else LexicalHit
        return await self._session.read(statement+' ORDER BY score DESC,'+key,params,row_type,limit)

    async def lexical_messages(self,tokens,*,phrase=False,candidate_ids=None,limit=200):
        return await self._lexical('messages',tokens,phrase=phrase,candidate_ids=candidate_ids,limit=limit)

    async def lexical_attachments(self,tokens,*,phrase=False,candidate_ids=None,limit=200):
        return await self._lexical('attachments',tokens,phrase=phrase,candidate_ids=candidate_ids,limit=limit)

    async def lexical_facts(self,tokens,*,phrase=False,limit=2000):
        return await self._lexical('facts',tokens,phrase=phrase,limit=limit)

    async def hydrate_embeddings(self,embedding_ids):
        ids=_ids(embedding_ids,numeric=True,maximum=10_000)
        return await self._session.read('''SELECT e.user_id AS owner_id,e.id,e.message_id,e.attachment_id,e.chunk_type,left(e.chunk_text,200) AS chunk_text,
            octet_length(e.chunk_text) AS chunk_bytes,(e.chunk_text IS NULL OR char_length(e.chunk_text)<=200) AS chunk_complete,
            m.thread_id,m.subject,m.from_addr,m.date,a.filename AS att_filename,e.model
            FROM public.embeddings e JOIN public.messages m ON m.user_id=e.user_id AND m.id=e.message_id
            LEFT JOIN public.attachments a ON a.user_id=e.user_id AND a.message_id=e.message_id AND a.id=e.attachment_id
            WHERE e.user_id=%s AND e.model=%s AND e.id=ANY(%s::bigint[]) ORDER BY e.id''',
            [self._session.owner_id,self._session.profile.embedding_tag,list(ids)],EmbeddingHit,max(1,len(ids)))

    async def hydrate_messages(self,message_ids,*,body_chars=200):
        _limit(body_chars,400_000)
        ids=_ids(message_ids,maximum=10_000)
        return await self._session.read('''SELECT m.user_id AS owner_id,m.id AS message_id,m.thread_id,m.subject,m.from_addr,m.to_addr,
            m.date,left(m.body_text,%s) AS body_text,octet_length(m.body_text) AS body_bytes,
            char_length(m.body_text)<=%s AS body_complete,m.labels,left(s.summary,16000) AS summary,
            octet_length(s.summary) AS summary_bytes,(s.summary IS NULL OR char_length(s.summary)<=16000) AS summary_complete,
            s.model AS summary_model,s.created_at AS summary_created_at FROM public.messages m
            LEFT JOIN LATERAL (SELECT ms.summary,ms.model,ms.created_at FROM public.message_summaries ms
                WHERE ms.user_id=m.user_id AND ms.message_id=m.id ORDER BY ms.created_at DESC,ms.model ASC LIMIT 1) s ON true
            WHERE m.user_id=%s AND m.id=ANY(%s::text[]) ORDER BY m.id''',[body_chars,body_chars,self._session.owner_id,list(ids)],MessageRow,max(1,len(ids)))

    async def hydrate_threads(self,thread_ids):
        ids=_ids(thread_ids,maximum=10_000)
        return await self._session.read('''SELECT user_id AS owner_id,thread_id,subject,participants,all_from_addrs,all_labels,
            date_first,date_last,message_count FROM public.thread_summary
            WHERE user_id=%s AND thread_id=ANY(%s::text[]) ORDER BY thread_id''',[self._session.owner_id,list(ids)],ThreadRow,max(1,len(ids)))

    async def owner_aliases(self,*,after=None,limit=1000):
        return await self._named_page('term_aliases','term','term,expansions',AliasRow,after,limit)

    async def owner_contacts(self,*,after=None,limit=1000):
        return await self._named_page('contact_frequency','email','email,score',ContactRow,after,limit)

    async def _named_page(self,table,key,columns,row_type,after,limit):
        _limit(limit,1000)
        if after is not None:_text(after)
        statement=f'SELECT user_id AS owner_id,{columns} FROM public.{table} WHERE user_id=%s'
        params=[self._session.owner_id]
        if after is not None:statement+=f' AND {key}>%s';params.append(after)
        return await self._session.read(statement+f' ORDER BY {key}',params,row_type,limit,cursor=key)

    async def fact_count(self):
        return await self._session.count('SELECT count(*) FROM public.propositions WHERE user_id=%s',[self._session.owner_id])

    async def fact_vectors_page(self,*,after=None,fact_ids=None,limit=128):
        return await self._vectors('propositions',FactVectorRow,after,fact_ids,None,limit)

    async def restricted_vectors_page(self,candidate_ids,*,after=None,limit=128):
        return await self._vectors('embeddings',EmbeddingVectorRow,after,None,_ids(candidate_ids),limit)

    async def _vectors(self,table,row_type,after,ids,message_ids,limit):
        _limit(limit,128)
        if after is not None:_ids((after,),numeric=True)
        if ids is not None:ids=_ids(ids,numeric=True,maximum=10_000)
        model=self._session.profile.fact_model_tag if table=='propositions' else self._session.profile.embedding_tag
        prefix='thread_id,text,' if table=='propositions' else ''
        # Guard BYTEA length before encoding/transferring it. Mismatch is explicit
        # metadata; callers decide coverage/error rather than silently rank zeros.
        statement=f'''SELECT user_id AS owner_id,id,message_id,{prefix}model,
            CASE WHEN model=%s AND octet_length(embedding)=%s THEN encode(embedding,'hex') ELSE NULL END AS embedding,
            CASE WHEN model<>%s THEN 'model_mismatch' WHEN embedding IS NULL THEN 'missing_vector'
                WHEN octet_length(embedding)<>%s THEN 'invalid_vector' ELSE 'ok' END AS vector_status
            FROM public.{table} WHERE user_id=%s'''
        params=[model,self._session.profile.dimensions*4,model,self._session.profile.dimensions*4,self._session.owner_id]
        if after is not None:statement+=' AND id>%s';params.append(after)
        if ids is not None:statement+=' AND id=ANY(%s::bigint[])';params.append(list(ids))
        if message_ids is not None:statement+=' AND message_id=ANY(%s::text[])';params.append(list(message_ids))
        return await self._session.read(statement+' ORDER BY id',params,row_type,limit,cursor='id')

    async def message_topics(self,message_ids):
        ids=_ids(message_ids,maximum=10_000)
        return await self._session.read('''SELECT mt.user_id AS owner_id,mt.message_id,mt.topic_id,t.label
            FROM public.message_topics mt JOIN public.topics t ON t.user_id=mt.user_id AND t.topic_id=mt.topic_id
            WHERE mt.user_id=%s AND mt.message_id=ANY(%s::text[]) ORDER BY mt.message_id,mt.topic_id''',
            [self._session.owner_id,list(ids)],TopicRow,10_000)

    async def topic_facets(self,message_ids,*,limit=1000):
        ids=_ids(message_ids);_limit(limit,1000)
        return await self._session.read('''SELECT mt.user_id AS owner_id,mt.topic_id,t.label,count(*) AS count
            FROM public.message_topics mt JOIN public.topics t ON t.user_id=mt.user_id AND t.topic_id=mt.topic_id
            WHERE mt.user_id=%s AND mt.message_id=ANY(%s::text[]) AND NOT EXISTS
                (SELECT 1 FROM public.topics child WHERE child.user_id=t.user_id AND child.parent_id=t.topic_id)
            GROUP BY mt.user_id,mt.topic_id,t.label ORDER BY count DESC,mt.topic_id''',
            [self._session.owner_id,list(ids)],TopicFacet,limit)
