"""Bounded, owner-scoped replay of untrusted agent output, not mailbox copies.

Events can contain private model output. Keep the registry private, render output
as untrusted content, and apply the deployment retention policy to terminal runs.
Browser adapters derive owner from authentication; guests append with a run token.
"""
import json

from .registry import AccessDenied

# text_delta: answer text streamed as generated; the matching `text` event still
# carries each complete message, so replay and the final answer ignore deltas.
_KINDS = {'text','text_delta','tool_start','tool_result','error','usage','status','artifact'}


class Events:
    def __init__(self, capabilities, *, max_events=10000, max_event_bytes=65536,
                 max_run_bytes=8*1024**2, max_owner_bytes=64*1024**2):
        values = (max_events,max_event_bytes,max_run_bytes,max_owner_bytes)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError('Positive replay limits required')
        self.capabilities,self.registry = capabilities,capabilities.registry
        self.max_events,self.max_event_bytes,self.max_run_bytes,self.max_owner_bytes = values
        with self.registry._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS run_events(run_id TEXT, seq INTEGER, body TEXT NOT NULL, PRIMARY KEY(run_id,seq))')
            db.execute('CREATE TABLE IF NOT EXISTS event_runs(run_id TEXT PRIMARY KEY, count INTEGER NOT NULL, bytes INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS event_owners(owner_id TEXT PRIMARY KEY, bytes INTEGER NOT NULL)')

    def append(self, token, event):
        self.capabilities.authorize(token,audience='events',operation='append')
        if (type(event) is not dict or type(event.get('type')) is not str or event['type'] not in _KINDS
                or event.keys() & {'owner_id','conversation_id','run_id','seq'}):
            raise AccessDenied()
        try:
            body=json.dumps(event,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            size=len(body.encode('utf-8'))
        except (ValueError,TypeError,RecursionError,UnicodeError):
            raise AccessDenied() from None
        if size>self.max_event_bytes:
            raise AccessDenied()
        with self.registry._transaction() as db:
            run=self.capabilities._authorize(db,token,'events','append')
            self.registry._fence(db,run)
            db.execute('INSERT OR IGNORE INTO event_runs VALUES(?,0,0)',(run['run_id'],))
            db.execute('INSERT OR IGNORE INTO event_owners VALUES(?,0)',(run['owner_id'],))
            usage=db.execute('SELECT count,bytes FROM event_runs WHERE run_id=?',(run['run_id'],)).fetchone()
            owner_bytes=db.execute('SELECT bytes FROM event_owners WHERE owner_id=?',(run['owner_id'],)).fetchone()[0]
            if usage['count']>=self.max_events or usage['bytes']+size>self.max_run_bytes or owner_bytes+size>self.max_owner_bytes:
                raise AccessDenied()
            seq=usage['count']+1
            db.execute('INSERT INTO run_events VALUES(?,?,?)',(run['run_id'],seq,body))
            db.execute('UPDATE event_runs SET count=?,bytes=bytes+? WHERE run_id=?',(seq,size,run['run_id']))
            db.execute('UPDATE event_owners SET bytes=bytes+? WHERE owner_id=?',(size,run['owner_id']))
            return seq

    def read(self, owner_id, conversation_id, run_id, *, after=0, limit=100):
        if type(after) is not int or after<0 or type(limit) is not int or not 1<=limit<=100:
            raise AccessDenied()
        with self.registry._transaction() as db:
            self.registry._owner(owner_id)
            if not db.execute('SELECT 1 FROM runs WHERE run_id=? AND owner_id=? AND conversation_id=?',
                              (run_id,owner_id,conversation_id)).fetchone():
                raise AccessDenied()
            rows=db.execute('SELECT seq,body FROM run_events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?',
                            (run_id,after,limit)).fetchall()
        self.registry._owner(owner_id)
        return [{'seq':row['seq'],'event':json.loads(row['body'])} for row in rows]

    def purge_terminal(self, owner_id, run_id):
        """Trusted retention/deletion hook; never exposed to a guest token."""
        with self.registry._transaction() as db:
            run=db.execute('SELECT status FROM runs WHERE run_id=? AND owner_id=?',(run_id,owner_id)).fetchone()
            if not run or run['status']=='active':
                raise AccessDenied()
            usage=db.execute('SELECT bytes FROM event_runs WHERE run_id=?',(run_id,)).fetchone()
            if usage:
                db.execute('UPDATE event_owners SET bytes=bytes-? WHERE owner_id=?',(usage['bytes'],owner_id))
            db.execute('DELETE FROM run_events WHERE run_id=?',(run_id,))
            db.execute('DELETE FROM event_runs WHERE run_id=?',(run_id,))
