"""Owned browser orchestration over the existing run, worker and event stores.

Trusted callbacks are bounded synchronous operations. ``persist_answer`` must
be idempotent by run ID and enforce authorization in its external commit. It
runs OUTSIDE SQLite; the post-check can withhold browser publication but cannot
undo an external commit whose acknowledgement races cancellation/revocation.
Input preparation owns its resources until the backend stop acknowledgement.
"""
import asyncio
import json
import re
import time
import uuid

from .event_http import _settled_call
from .maintenance import require_ready_in
from .registry import AccessDenied

_TERMINAL=frozenset(('completed','failed','cancelled'))
# Guest agents a run may boot; see full_agent_remote.GUEST_PROFILES.
RUNTIMES=frozenset(('pi','pi_gemini','pi_opus','claude'))


def _conversation(value):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}',value):
        raise ValueError('Conversation required')


async def _wait(task):
    """Retain ownership through repeated caller cancellation; report it last."""
    interrupted=False
    while not task.done():
        try:await asyncio.shield(task)
        except asyncio.CancelledError:interrupted=True
        except Exception:break
    return interrupted


class BrowserRuns:
    def __init__(self,workers,events,*,prepare_input,persist_answer,budget_for,poll_seconds=.25):
        if workers.registry is not events.registry:
            raise ValueError('Browser runs must share the worker event registry')
        if not all(callable(item) for item in (prepare_input,persist_answer,budget_for)):
            raise ValueError('Trusted runtime, persistence and budget composition required')
        if type(poll_seconds) not in (int,float) or not .01<=poll_seconds<=1:
            raise ValueError('Bounded event polling interval required')
        self.workers,self.events,self.registry=workers,events,workers.registry
        self.prepare_input,self.persist_answer,self.budget_for=prepare_input,persist_answer,budget_for
        self.poll_seconds=poll_seconds
        self._tasks={}
        self._admissions=set()
        self._cancellations=set()
        self._cancel_locks={}
        self._worker_lock=asyncio.Lock()
        self._runs=set()
        self._closed=False
        self._close_task=None
        with self.registry._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS browser_runs (
                run_id TEXT PRIMARY KEY, state TEXT NOT NULL,
                answer TEXT NOT NULL DEFAULT '')''')

    async def _worker(self,method,run_id):
        # WorkerController uses a nonblocking process-wide file lock. Avoid
        # manufacturing contention between this adapter's own healthy runs.
        # Durable revocation happens before waiting here on cancellation.
        async with self._worker_lock:
            return await _settled_call(method,run_id)

    def _open(self,owner,conversation):
        _conversation(conversation)
        budget=self.budget_for(owner)
        # Keep the host deadline inside the remote worker ceiling. An exact
        # boundary incorrectly rejects admission under ordinary clock skew.
        ttl=max(1,min(1800,self.workers.limits.wall_seconds)-5)
        lease=self.registry.start_run(owner,conversation,request_key=uuid.uuid4().hex,
            budget_id=budget,deadline_ttl=ttl)
        try:
            with self.registry._transaction() as db:
                self.registry._active(db,lease.run_id)
                db.execute("INSERT INTO browser_runs(run_id,state) VALUES(?,'starting')",(lease.run_id,))
        except BaseException:
            self.registry.cancel(lease.run_id)
            raise
        return lease

    async def _admit(self,owner,conversation,question,runtime):
        lease=await asyncio.to_thread(self._open,owner,conversation)
        self._runs.add(lease.run_id)
        if self._closed:
            await _settled_call(self.registry.cancel,lease.run_id)
            await _settled_call(self._state,lease.run_id,'cancelled')
            raise AccessDenied()
        entered=asyncio.Event()
        task=asyncio.create_task(self._drive(lease,question,entered,runtime))
        self._tasks[lease.run_id]=task
        def done(finished):
            self._tasks.pop(lease.run_id,None)
            if not finished.cancelled():finished.exception()
        task.add_done_callback(done)
        # close drains admissions before cancelling drives. Callers never obtain
        # a cancellable drive which has not entered its finally scope.
        await entered.wait()
        return lease.run_id

    async def start(self,owner,conversation,question,runtime='pi'):
        if self._closed:raise AccessDenied()
        if runtime not in RUNTIMES:raise ValueError('Unsupported agent runtime')
        if (type(question) is not str or not question.strip() or '\x00' in question
                or len(question.encode('utf-8'))>16384):
            raise ValueError('Question must contain at most 16 KiB of text')
        admission=asyncio.create_task(self._admit(owner,conversation,question,runtime))
        self._admissions.add(admission)
        try:
            interrupted=await _wait(admission)
            try:run_id=admission.result()
            except BaseException:
                if interrupted:raise asyncio.CancelledError() from None
                raise
            if interrupted:
                # The browser may vanish after a successful admission. Drain
                # cancellation without requiring owner authorization again.
                cleanup=asyncio.create_task(self._cancel(run_id))
                await _wait(cleanup)
                cleanup.result()
                raise asyncio.CancelledError()
            return run_id
        finally:self._admissions.discard(admission)

    def _state(self,run_id,state,answer=''):
        with self.registry._transaction() as db:
            if state=='running':
                db.execute("UPDATE browser_runs SET state='running' WHERE run_id=? AND state='starting'",(run_id,))
            else:
                db.execute("UPDATE browser_runs SET state=?,answer=? WHERE run_id=? AND state NOT IN ('completed','failed','cancelled')",
                    (state,answer,run_id))

    def _authorize(self,db,owner,conversation,run_id):
        require_ready_in(db,self.registry._release_identity)
        row=db.execute('''SELECT r.*,b.state,b.answer FROM browser_runs b JOIN runs r USING(run_id)
            WHERE r.run_id=? AND r.owner_id=? AND r.conversation_id=?''',(run_id,owner,conversation)).fetchone()
        if row is None:raise AccessDenied()
        identity=self.registry._release_identity
        if identity is not None and row['release_epoch']!=identity.release_epoch:raise AccessDenied()
        self.registry._owner(owner)
        return row

    def _active(self,run_id):
        with self.registry._transaction() as db:
            self.registry._fence(db,self.registry._active(db,run_id))

    def _complete_run(self,run_id):
        with self.registry._transaction() as db:
            row=self.registry._active(db,run_id)
            self.registry._fence(db,row)
            state=db.execute('SELECT state FROM browser_runs WHERE run_id=?',(run_id,)).fetchone()[0]
            if state not in ('starting','running'):raise AccessDenied()
            self.registry._finish(db,run_id,'completed')

    def _publish_check(self,db,lease):
        row=self._authorize(db,lease.owner_id,lease.conversation_id,lease.run_id)
        if (self._closed or row['state'] not in ('starting','running') or row['status']!='completed'
                or min(row['deadline'],row['lease_expires'])<=self.registry.clock()):raise AccessDenied()
        self.registry._fence(db,row)
        worker=db.execute('SELECT state FROM workers WHERE run_id=?',(lease.run_id,)).fetchone()
        if worker is None or worker['state']!='stopped':raise AccessDenied()
        return row

    def _publish(self,lease,answer):
        with self.registry._transaction() as db:self._publish_check(db,lease)
        # No SQLite transaction spans the externally owned persistence call.
        if self.persist_answer(lease.owner_id,lease.conversation_id,lease.run_id,answer) is not True:
            raise AccessDenied()
        with self.registry._transaction() as db:
            self._publish_check(db,lease)
            db.execute("UPDATE browser_runs SET state='completed',answer=? WHERE run_id=?",(answer,lease.run_id))

    def _revoke(self,run_id,desired):
        with self.registry._transaction() as db:
            row=db.execute('SELECT state FROM browser_runs WHERE run_id=?',(run_id,)).fetchone()
            status='failed' if desired=='failed' and row['state']!='cancelling' else 'cancelled'
            if desired!='completed':self.registry._finish(db,run_id,status)

    def _outcome(self,run_id,desired):
        with self.registry._transaction() as db:
            row=db.execute('SELECT b.state,r.status FROM browser_runs b JOIN runs r USING(run_id) WHERE run_id=?',(run_id,)).fetchone()
            if row['state'] in _TERMINAL:return row['state']
            if row['state']=='cancelling':desired='cancelled'
            if row['status']=='active':self.registry._finish(db,run_id,'failed' if desired=='failed' else 'cancelled')
            db.execute('UPDATE browser_runs SET state=?,answer=? WHERE run_id=?',(desired,'',run_id))
            return desired

    async def _drive(self,lease,question,entered,runtime='pi'):
        desired='failed'
        answer=''
        try:
            entered.set()
            await _settled_call(self.prepare_input,lease,question,runtime)
            await self._worker(self.workers.start,lease.run_id)
            await _settled_call(self._state,lease.run_id,'running')
            cursor,heartbeat=0,time.monotonic()
            while True:
                await _settled_call(self._active,lease.run_id)
                batch=await _settled_call(self.events.read,lease.owner_id,lease.conversation_id,lease.run_id,after=cursor)
                completed=False
                for row in batch:
                    cursor=row['seq'];event=row['event']
                    if event['type']=='text':
                        text=event.get('text')
                        if type(text) is not str or len((answer+text).encode('utf-8'))>65536:
                            raise ValueError('Invalid or oversized worker answer')
                        answer+=text
                    if event['type']=='status' and event.get('state')=='runner_completed':
                        completed=True
                        break
                if completed:
                    if not answer.strip():raise ValueError('Worker returned no answer')
                    await _settled_call(self._complete_run,lease.run_id)
                    desired='completed'
                    break
                if time.monotonic()-heartbeat>=10:
                    await self._worker(self.workers.heartbeat,lease.run_id)
                    heartbeat=time.monotonic()
                await asyncio.sleep(self.poll_seconds)
        except asyncio.CancelledError:desired='cancelled'
        except Exception:desired='failed'
        finally:
            async def cleanup():
                nonlocal desired
                try:
                    await _settled_call(self._revoke,lease.run_id,desired)
                    await self._worker(self.workers.cancel,lease.run_id)
                except Exception:
                    await _settled_call(self._state,lease.run_id,'stopping')
                    return
                if desired=='completed':
                    try:
                        await _settled_call(self._publish,lease,answer)
                        return
                    except Exception:desired='failed'
                await _settled_call(self._outcome,lease.run_id,desired)
            cleanup_task=asyncio.create_task(cleanup())
            await _wait(cleanup_task)
            cleanup_task.result()

    def snapshot(self,owner,conversation,run_id,*,after=0):
        _conversation(conversation)
        with self.registry._transaction() as db:
            row=self._authorize(db,owner,conversation,run_id)
            state,answer=row['state'],row['answer']
        events=self.events.read(owner,conversation,run_id,after=after)
        # Completion can commit between these reads. Keep the earlier state so
        # SSE cannot terminate using an event page fetched before completion.
        with self.registry._transaction() as db:self._authorize(db,owner,conversation,run_id)
        return dict(state=state,answer=answer,events=events)

    def _request_cancel(self,run_id,owner=None,conversation=None):
        with self.registry._transaction() as db:
            if owner is not None:row=self._authorize(db,owner,conversation,run_id)
            else:row=db.execute('SELECT state FROM browser_runs WHERE run_id=?',(run_id,)).fetchone()
            if row is None:raise AccessDenied()
            if row['state'] in _TERMINAL:return False
            db.execute("UPDATE browser_runs SET state='cancelling',answer='' WHERE run_id=?",(run_id,))
            self.registry._finish(db,run_id,'cancelled')
            return True

    async def _cancel(self,run_id,owner=None,conversation=None):
        lock=self._cancel_locks.setdefault(run_id,asyncio.Lock())
        async with lock:
            needed=await _settled_call(self._request_cancel,run_id,owner,conversation)
            if not needed:return
            task=self._tasks.get(run_id)
            if task is not None:
                task.cancel()
                await _wait(task)
                if not task.cancelled():task.result()
            # Covers completed tasks, restart recovery and a defensive immediate
            # cancellation of a drive before its body executes.
            try:await self._worker(self.workers.cancel,run_id)
            except Exception:
                await _settled_call(self._state,run_id,'stopping')
                raise AccessDenied() from None
            await _settled_call(self._outcome,run_id,'cancelled')

    async def _owned_cancel(self,run_id,owner=None,conversation=None):
        task=asyncio.create_task(self._cancel(run_id,owner,conversation))
        self._cancellations.add(task)
        try:
            interrupted=await _wait(task)
            task.result()
            if interrupted:raise asyncio.CancelledError()
        finally:self._cancellations.discard(task)

    async def cancel(self,owner,conversation,run_id):
        _conversation(conversation)
        await self._owned_cancel(run_id,owner,conversation)

    async def abandon_unpublished(self,run_id):
        """Trusted HTTP-admission compensation, never a caller-selected route.

        Only pass the ID just returned by this instance's start(), when its
        authenticated response cannot be published. Current owner revocation
        must not prevent resource teardown. Completed answers remain terminal.
        """
        if type(run_id) is not str or run_id not in self._runs:raise AccessDenied()
        await self._owned_cancel(run_id)

    async def recover(self):
        """Single-process service startup: stop abandoned runs before admission.

        Call only before accepting requests. A failed cleanup prevents startup;
        its durable stopping binding remains available to a later retry.
        """
        if self._tasks or self._admissions or self._closed:
            raise AccessDenied()
        def pending():
            with self.registry._transaction() as db:
                return [row[0] for row in db.execute(
                    "SELECT run_id FROM browser_runs WHERE state NOT IN ('completed','failed','cancelled')")]
        failed = False
        for run_id in await _settled_call(pending):
            try:
                await self._owned_cancel(run_id)
            except AccessDenied:
                failed = True
        async with self._worker_lock:
            try:
                await _settled_call(self.workers.reconcile)
            except AccessDenied:
                failed = True
        if failed:
            raise AccessDenied()

    async def _close(self):
        # _closed was set before taking this snapshot: no new admissions can
        # appear. Each admission starts its drive cleanup scope before return.
        await asyncio.gather(*tuple(self._admissions),return_exceptions=True)
        await asyncio.gather(*tuple(self._cancellations),return_exceptions=True)
        results=await asyncio.gather(*(self._cancel(run_id) for run_id in tuple(self._runs)),return_exceptions=True)
        await asyncio.gather(*tuple(self._tasks.values()),return_exceptions=True)
        if any(isinstance(result,BaseException) for result in results):raise AccessDenied()

    async def close(self):
        self._closed=True
        if self._close_task is None:self._close_task=asyncio.create_task(self._close())
        interrupted=await _wait(self._close_task)
        self._close_task.result()
        if interrupted:raise asyncio.CancelledError()

def event_frame(row):
    """Existing browser SSE envelope; IDs/timing come from the trusted store."""
    event = row['event']
    kind = {'tool_start':'tool_call','tool_result':'tool_result',
        'text':'writer','status':'status','artifact':'artifact','usage':'usage','error':'error'}[event['type']]
    return f'event: {kind}\ndata: '+json.dumps(
        {'seq':row['seq'],'agent':'agent','payload':event},ensure_ascii=False)+'\n\n'
