"""Trusted persistence into existing shared PostgreSQL conversation tables.

No mailbox data is copied. ``install`` is an explicit deployment migration for
one small idempotency table; it is never called by request handling. The factory
must supply fresh trusted app-writer connections, never guest reader credentials.
Authorization runs outside Registry transactions and immediately before the PG
commit. Cross-store revocation and PG COMMIT are not an atomic transaction.
"""
from contextlib import contextmanager
import hashlib
import json
import re

from psycopg.pq import TransactionStatus

from .registry import AccessDenied


_MAX_INPUT_MESSAGES = 100
_MAX_USER_TURN_BYTES = 16 * 1024
_MAX_READ_MESSAGES = 1000
_MAX_READ_PARTS_BYTES = 64 * 1024**2


def validate_receipt_schema(db):
    """Validate the receipt migration contract before installing writer privileges."""
    relation=db.execute("SELECT oid,relkind FROM pg_class WHERE oid=to_regclass('public.browser_answer_receipts')").fetchone()
    if not relation or relation[1]!='r':
        raise ValueError('Incompatible browser receipt schema')
    columns=db.execute('''SELECT a.attname,a.atttypid='pg_catalog.text'::regtype,a.attnotnull,a.attnum,
            a.atthasdef,a.attgenerated,a.attidentity FROM pg_attribute a
        WHERE a.attrelid=%s AND a.attnum>0 AND NOT a.attisdropped''',(relation[0],)).fetchall()
    if (len(columns)!=4 or {r[0] for r in columns}!={'run_id','owner_id','conversation_id','content_hash'}
            or any(not r[1] or not r[2] or r[4] or r[5] or r[6] for r in columns)):
        raise ValueError('Incompatible browser receipt schema')
    numbers={r[0]:r[3] for r in columns}
    parent=db.execute("SELECT attnum FROM pg_attribute WHERE attrelid='public.conversations'::regclass AND attname='id' AND NOT attisdropped").fetchone()
    constraints=db.execute('''SELECT contype,conkey,confkey,confdeltype,
        confrelid='public.conversations'::regclass,convalidated,condeferrable
        FROM pg_constraint WHERE conrelid=%s''',(relation[0],)).fetchall()
    expected_primary=('p',[numbers['run_id']],None,' ',False,True,False)
    expected_foreign=('f',[numbers['conversation_id']],[parent[0]] if parent else [],'c',True,True,False)
    if len(constraints)!=2 or expected_primary not in constraints or expected_foreign not in constraints:
        raise ValueError('Incompatible browser receipt schema')


class BrowserConversations:
    @staticmethod
    def install(db):
        with db.transaction():
            db.execute("SELECT pg_advisory_xact_lock(hashtextextended('gms_browser_receipt_schema',0))")
            db.execute('''CREATE TABLE IF NOT EXISTS public.browser_answer_receipts (
                run_id text PRIMARY KEY, owner_id text NOT NULL,
                conversation_id text NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
                content_hash text NOT NULL)''')
            db.execute('LOCK TABLE public.browser_answer_receipts IN ACCESS EXCLUSIVE MODE')
            validate_receipt_schema(db)
            db.execute('REVOKE ALL ON public.browser_answer_receipts FROM PUBLIC')
            db.execute('ALTER TABLE public.browser_answer_receipts ENABLE ROW LEVEL SECURITY')

    def __init__(self,connect,*,is_active,authorize_persistence,read_events,can_edit=None,transaction_factory=None):
        if (not all(callable(item) for item in (is_active,authorize_persistence,read_events))
                or not (callable(connect) if transaction_factory is None else callable(transaction_factory))):
            raise ValueError('Trusted persistence composition required')
        self.connect,self.is_active = connect,is_active
        self.authorize_persistence,self.read_events = authorize_persistence,read_events
        self.can_edit = can_edit
        self.transaction_factory = transaction_factory

    def _identity(self,owner):
        if type(owner) is not str or not 1<=len(owner)<=256:
            raise AccessDenied()
        try:
            active = self.is_active(owner)
        except Exception:
            raise AccessDenied() from None
        if active is not True:
            raise AccessDenied()

    def _owner(self,owner,conversation):
        self._identity(owner)
        if (type(conversation) is not str
                or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}',conversation)):
            raise AccessDenied()

    def _editable(self,owner,conversation):
        if not callable(self.can_edit):
            raise AccessDenied()
        try:
            allowed = self.can_edit(owner,conversation)
        except Exception:
            raise AccessDenied() from None
        if allowed is not True:
            raise AccessDenied()

    @contextmanager
    def _connection(self,owner):
        if self.transaction_factory is not None:
            with self.transaction_factory(owner) as db:
                db.execute("SET LOCAL search_path = pg_catalog, public")
                db.execute("SET LOCAL statement_timeout = '5s'")
                db.execute("SET LOCAL lock_timeout = '2s'")
                yield db
            return
        db = self.connect()
        with db:
            if db.autocommit or db.info.transaction_status != TransactionStatus.IDLE:
                raise AccessDenied()
            db.execute("SET LOCAL search_path = pg_catalog, public")
            db.execute("SET LOCAL statement_timeout = '5s'")
            db.execute("SET LOCAL lock_timeout = '2s'")
            yield db

    def claim(self,owner,conversation):
        self._owner(owner,conversation)
        with self._connection(owner) as db:
            db.execute('''INSERT INTO public.conversations(id,user_id) VALUES(%s,%s)
                ON CONFLICT(id) DO NOTHING''',(conversation,owner))
            row = db.execute('SELECT user_id FROM public.conversations WHERE id=%s FOR UPDATE',
                (conversation,)).fetchone()
            if row is None or row[0] != owner:
                raise AccessDenied()
            self._owner(owner,conversation)
        return True

    def list(self,owner,limit=100):
        self._identity(owner)
        if type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Conversation limit must be between 1 and 100')
        with self._connection(owner) as db:
            rows = db.execute('''SELECT c.id,COALESCE(c.title,'New chat'),
                    c.created_at,c.updated_at,
                    (SELECT COUNT(*) FROM public.conversation_messages m
                     WHERE m.conversation_id=c.id)
                FROM public.conversations c WHERE c.user_id=%s
                ORDER BY c.updated_at DESC,c.id LIMIT %s''',(owner,limit)).fetchall()
            self._identity(owner)
        return [dict(id=row[0],title=row[1],created_at=row[2],updated_at=row[3],
            message_count=row[4]) for row in rows]

    def get(self,owner,conversation):
        self._owner(owner,conversation)
        with self._connection(owner) as db:
            row = db.execute('''SELECT id,title,created_at,updated_at
                FROM public.conversations WHERE id=%s AND user_id=%s FOR SHARE''',
                (conversation,owner)).fetchone()
            if row is None:
                self._identity(owner)
                return None
            size = db.execute('''SELECT COUNT(*),COALESCE(SUM(octet_length(parts)),0)
                FROM (SELECT parts FROM public.conversation_messages
                      WHERE conversation_id=%s ORDER BY seq LIMIT %s) bounded''',
                (conversation,_MAX_READ_MESSAGES+1)).fetchone()
            if size[0] > _MAX_READ_MESSAGES or size[1] > _MAX_READ_PARTS_BYTES:
                raise AccessDenied()
            message_rows = db.execute('''SELECT seq,role,parts
                FROM public.conversation_messages WHERE conversation_id=%s
                ORDER BY seq LIMIT %s''',(conversation,_MAX_READ_MESSAGES+1)).fetchall()
            messages = []
            try:
                for seq,role,parts in message_rows:
                    value = json.loads(parts)
                    if type(value) is not list:
                        raise ValueError()
                    messages.append(dict(seq=seq,role=role,parts=value))
            except (TypeError,ValueError,UnicodeError,RecursionError):
                raise AccessDenied() from None
            self._identity(owner)
        return dict(id=row[0],title=row[1],created_at=row[2],updated_at=row[3],
            messages=messages)

    @staticmethod
    def _user_messages(payload):
        if type(payload) is not dict or set(payload)-{'title','messages'}:
            raise AccessDenied()
        title = payload.get('title')
        if title is not None and (type(title) is not str or len(title)>200 or '\x00' in title):
            raise AccessDenied()
        try:
            if title is not None:
                title.encode('utf-8')
        except UnicodeError:
            raise AccessDenied() from None
        incoming = payload.get('messages',[])
        if type(incoming) is not list or len(incoming)>_MAX_INPUT_MESSAGES:
            raise AccessDenied()
        turns = []
        for message in incoming:
            if (type(message) is not dict or set(message)!=set(('role','parts'))
                    or message.get('role')!='user' or type(message.get('parts')) is not list
                    or not 1<=len(message['parts'])<=100):
                raise AccessDenied()
            texts = []
            for part in message['parts']:
                if (type(part) is not dict or set(part)!=set(('type','text'))
                        or part.get('type')!='text' or type(part.get('text')) is not str
                        or '\x00' in part['text']):
                    raise AccessDenied()
                texts.append(part['text'])
            try:
                if not ''.join(texts).strip() or sum(len(text.encode('utf-8')) for text in texts)>_MAX_USER_TURN_BYTES:
                    raise AccessDenied()
                encoded = json.dumps(message['parts'],ensure_ascii=False,allow_nan=False,
                    separators=(',',':'))
            except (TypeError,ValueError,UnicodeError):
                raise AccessDenied() from None
            turns.append((tuple(texts),encoded))
        return title,turns

    @staticmethod
    def _stored_user_turn(parts):
        try:
            value = json.loads(parts)
            if type(value) is not list or not 1<=len(value)<=100:
                raise ValueError()
            texts = []
            for part in value:
                if (type(part) is not dict or part.get('type')!='text'
                        or type(part.get('text')) is not str):
                    raise ValueError()
                texts.append(part['text'])
            return tuple(texts)
        except (TypeError,ValueError,UnicodeError,RecursionError):
            raise AccessDenied() from None

    def save(self,owner,conversation,payload):
        self._owner(owner,conversation)
        title,incoming = self._user_messages(payload)
        self._editable(owner,conversation)
        with self._connection(owner) as db:
            db.execute('''INSERT INTO public.conversations(id,user_id,title)
                VALUES(%s,%s,%s) ON CONFLICT(id) DO NOTHING''',(conversation,owner,title))
            row = db.execute('''SELECT id FROM public.conversations
                WHERE id=%s AND user_id=%s FOR UPDATE''',(conversation,owner)).fetchone()
            if row is None:
                raise AccessDenied()
            stored = db.execute('''SELECT parts FROM public.conversation_messages
                WHERE conversation_id=%s AND role='user' ORDER BY seq LIMIT %s''',
                (conversation,_MAX_INPUT_MESSAGES+1)).fetchall()
            if len(stored)>_MAX_INPUT_MESSAGES:
                raise AccessDenied()
            prefix = [self._stored_user_turn(item[0]) for item in stored]
            supplied = [item[0] for item in incoming]
            if supplied[:len(prefix)] != prefix or len(supplied) not in (len(prefix),len(prefix)+1):
                raise AccessDenied()
            if len(supplied)>len(prefix):
                db.execute('''INSERT INTO public.conversation_messages
                    (conversation_id,seq,role,parts)
                    SELECT %s,COALESCE(MAX(seq),-1)+1,'user',%s
                    FROM public.conversation_messages WHERE conversation_id=%s''',
                    (conversation,incoming[-1][1],conversation))
            if title is None:
                db.execute('''UPDATE public.conversations SET updated_at=NOW()
                    WHERE id=%s AND user_id=%s''',(conversation,owner))
            else:
                db.execute('''UPDATE public.conversations SET title=%s,updated_at=NOW()
                    WHERE id=%s AND user_id=%s''',(title,conversation,owner))
            self._owner(owner,conversation)
            self._editable(owner,conversation)
        return True

    def delete(self,owner,conversation):
        self._owner(owner,conversation)
        self._editable(owner,conversation)
        with self._connection(owner) as db:
            row = db.execute('''SELECT id FROM public.conversations
                WHERE id=%s AND user_id=%s FOR UPDATE''',(conversation,owner)).fetchone()
            if row is None:
                raise AccessDenied()
            db.execute('DELETE FROM public.conversations WHERE id=%s AND user_id=%s',
                (conversation,owner))
            self._owner(owner,conversation)
            self._editable(owner,conversation)
        return True

    def _parts(self,owner,conversation,run,answer):
        if (type(run) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}',run)
                or type(answer) is not str or not answer.strip()
                or len(answer.encode('utf-8')) > 65536):
            raise AccessDenied()
        parts, size, last = [], 0, 0
        mapping = {'tool_start':'tool_call','tool_result':'tool_result','text':'writer',
            'status':'status','usage':'usage','error':'error','artifact':'artifact'}
        for row in self.read_events(owner,conversation,run):
            if len(parts) >= 10000:
                raise AccessDenied()
            seq,event = row['seq'],row['event']
            if type(seq) is int and seq>last and event.get('type')=='text_delta':
                last = seq
                continue  # Live streaming only; the `text` event holds the message.
            if type(seq) is not int or seq<=last or event.get('type') not in mapping:
                raise AccessDenied()
            last = seq
            part = {'type':'data-deep-stage','id':f'run-{run}-{seq}',
                'data':{'kind':mapping[event['type']],'payload':event}}
            encoded = json.dumps(part,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            size += len(encoded.encode('utf-8'))
            if size > 10*1024**2:
                raise AccessDenied()
            parts.append(part)
        parts.append({'type':'text','text':answer})
        return json.dumps(parts,ensure_ascii=False,allow_nan=False,separators=(',',':'))

    def persist(self,owner,conversation,run,answer):
        self._owner(owner,conversation)
        if self.authorize_persistence(owner,conversation,run) is not True:
            raise AccessDenied()
        parts = self._parts(owner,conversation,run,answer)
        digest = hashlib.sha256(parts.encode('utf-8')).hexdigest()
        with self._connection(owner) as db:
            # This row lock serializes append sequence allocation and deletion.
            row = db.execute('''SELECT id FROM public.conversations
                WHERE id=%s AND user_id=%s FOR UPDATE''',(conversation,owner)).fetchone()
            if row is None:
                raise AccessDenied()
            existing = db.execute('''SELECT owner_id,conversation_id,content_hash
                FROM public.browser_answer_receipts WHERE run_id=%s''',(run,)).fetchone()
            if existing is not None:
                if tuple(existing) != (owner,conversation,digest):
                    raise AccessDenied()
            else:
                db.execute('''INSERT INTO public.browser_answer_receipts
                    (run_id,owner_id,conversation_id,content_hash) VALUES(%s,%s,%s,%s)''',
                    (run,owner,conversation,digest))
                db.execute('''INSERT INTO public.conversation_messages(conversation_id,seq,role,parts)
                    SELECT %s,COALESCE(MAX(seq),-1)+1,'assistant',%s
                    FROM public.conversation_messages WHERE conversation_id=%s''',
                    (conversation,parts,conversation))
                db.execute('UPDATE public.conversations SET updated_at=NOW() WHERE id=%s AND user_id=%s',
                    (conversation,owner))
            self._owner(owner,conversation)
            if self.authorize_persistence(owner,conversation,run) is not True:
                raise AccessDenied()
        return True


def compose_browser_runs(workers,events,*,connect,is_active,prepare_input,budget_for,poll_seconds=.25,transaction_factory=None):
    """Wire real transcript persistence to controller authorization and events.

    All infrastructure inputs are explicit. This neither installs tables nor
    changes public startup. The returned conversation adapter supplies the
    invited run router's atomic ``claim_conversation`` callback.
    """
    from .browser_runs import BrowserRuns

    def authorize(owner,conversation,run):
        with workers.registry._transaction() as db:
            row = runs._authorize(db,owner,conversation,run)
            runs._publish_check(db,workers.registry._lease(row))
        return True

    def read_events(owner,conversation,run):
        after = 0
        for _ in range(100):
            page = events.read(owner,conversation,run,after=after)
            yield from page
            if len(page) < 100:
                return
            after = page[-1]['seq']
        if events.read(owner,conversation,run,after=after):
            raise AccessDenied()

    def can_edit(owner,conversation):
        from .maintenance import require_ready_in
        registry = workers.registry
        with registry._transaction() as db:
            require_ready_in(db,registry._release_identity)
            registry._owner(owner)
            active = db.execute('''SELECT 1 FROM browser_runs b JOIN runs r USING(run_id)
                WHERE r.owner_id=? AND r.conversation_id=?
                  AND b.state NOT IN ('completed','failed','cancelled') LIMIT 1''',
                (owner,conversation)).fetchone()
        return active is None

    conversations = BrowserConversations(connect,is_active=is_active,
        authorize_persistence=authorize,read_events=read_events,can_edit=can_edit,
        transaction_factory=transaction_factory)
    def prepare_with_history(lease,question,runtime='pi'):
        from .browser_prompt import build_prompt
        saved = conversations.get(lease.owner_id,lease.conversation_id)
        if saved is None:
            raise AccessDenied()
        prepare_input(lease,build_prompt(saved['messages'],question),runtime)

    runs = BrowserRuns(workers,events,prepare_input=prepare_with_history,
        persist_answer=conversations.persist,budget_for=budget_for,poll_seconds=poll_seconds)
    return runs,conversations
