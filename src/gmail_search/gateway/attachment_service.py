"""Revocable attachment parser jobs bound to a run capability.

The trusted asynchronous loader receives the owner derived from the attachment
capability. It must cooperate with cancellation: accepting a synchronous loader
would leave an executor thread holding private mail bytes. A backend job's
``stop`` method must stop and reap its complete VM before it returns.
"""
import asyncio
import inspect

from .attachment_sandbox import AttachmentDenied, AttachmentInput, INPUT_BYTES, MIMES, validate_request, validate_result


class RunAttachmentService:
    def __init__(self, capabilities, backend, *, load, watch_interval=.05, max_jobs=2, max_owner_jobs=1):
        if (not inspect.iscoroutinefunction(load) or type(watch_interval) not in (int, float)
                or not 0 < watch_interval <= 1 or type(max_jobs) is not int
                or type(max_owner_jobs) is not int or not 1 <= max_owner_jobs <= max_jobs <= 8):
            raise AttachmentDenied()
        self.capabilities = capabilities
        self.backend = backend
        self.load = load
        self.watch_interval = watch_interval
        self.max_jobs, self.max_owner_jobs = max_jobs, max_owner_jobs
        self.active_jobs = {}

    async def _authorize(self, token):
        return await asyncio.to_thread(
            self.capabilities.authorize, token, audience='attachment', operation='parse')

    async def _watch(self, token):
        while True:
            await asyncio.sleep(self.watch_interval)
            await self._authorize(token)

    @staticmethod
    def _source(source, owner_id, attachment_id):
        if (type(source) is not AttachmentInput or source.owner_id != owner_id
                or source.attachment_id != attachment_id or source.mime_type not in MIMES
                or type(source.data) is not bytes or not 0 < len(source.data) <= INPUT_BYTES):
            raise AttachmentDenied()
        return source

    async def _await_authorized(self, task, watch):
        done, _ = await asyncio.wait((task, watch), return_when=asyncio.FIRST_COMPLETED)
        if watch in done:
            await watch
        return await task

    @staticmethod
    async def _drain(task):
        """Await a cleanup task despite repeated caller cancellation."""
        interrupted = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                # Keep the task reference and drain the cleanup acknowledgement
                # first. A later cancellation cannot strand private bytes/VMs.
                interrupted = True
        if interrupted:
            raise asyncio.CancelledError()

    async def _cleanup(self, job, load_task, job_task, watch):
        """One non-cancelled owner for VM, loader, and executor teardown."""
        for task in (load_task, watch):
            if task is not None and not task.done():
                task.cancel()
        try:
            if job is not None:
                # The job contract joins its worker after stopping/reaping the
                # whole VM. Do not cancel job_task: wait() is released by stop.
                await asyncio.to_thread(job.stop)
            elif job_task is not None and not job_task.done():
                job_task.cancel()
        finally:
            await asyncio.gather(*(task for task in (load_task, job_task, watch) if task is not None), return_exceptions=True)

    async def parse(self, token, attachment_id, *, dpi=100, pages=()):
        """Return one result only while the token remains authorized.

        ``owner_id`` is deliberately absent: it is always derived from the
        capability, so callers cannot select another mailbox.
        """
        validate_request(attachment_id, dpi=dpi, pages=pages)
        lease = await self._authorize(token)
        # Single gateway event loop/process. Reserve before even the metadata
        # lookup/file read, so bytes cannot pile up ahead of the worker's lock.
        if (sum(self.active_jobs.values()) >= self.max_jobs
                or self.active_jobs.get(lease.owner_id, 0) >= self.max_owner_jobs):
            raise AttachmentDenied()
        self.active_jobs[lease.owner_id] = self.active_jobs.get(lease.owner_id, 0) + 1
        watch = asyncio.create_task(self._watch(token))
        load_task = job_task = None
        job = None
        cleanup = None
        try:
            load_task = asyncio.create_task(self.load(lease.owner_id, attachment_id))
            source = self._source(await self._await_authorized(load_task, watch), lease.owner_id, attachment_id)
            lease = await self._authorize(token)
            options = {'dpi': dpi, 'pages': list(pages)}
            if callable(getattr(self.backend, 'start_authorized', None)):
                # A remote backend needs immutable context and online renewal
                # authorization. The callback and capability never leave host.
                job = self.backend.start_authorized(source.data, source.mime_type, options,
                    lease=lease, attachment_id=attachment_id,
                    authorize=lambda: self.capabilities.authorize(token, audience='attachment', operation='parse'))
            else:
                job = self.backend.start(source.data, source.mime_type, options)
            if not callable(getattr(job, 'wait', None)) or not callable(getattr(job, 'stop', None)):
                raise AttachmentDenied()
            job_task = asyncio.create_task(asyncio.to_thread(job.wait))
            raw = await self._await_authorized(job_task, watch)
            await self._authorize(token)
            result = validate_result(raw)
            cleanup = asyncio.create_task(self._cleanup(job, load_task, job_task, watch))
            await self._drain(cleanup)
            # Teardown can take seconds. Check after it, immediately before
            # result publication, so a late revocation cannot leak output.
            await self._authorize(token)
            return result
        finally:
            if cleanup is None:
                # This acknowledgement is the teardown boundary: do not return
                # output, denial, or cancellation while the VM remains live.
                cleanup = asyncio.create_task(self._cleanup(job, load_task, job_task, watch))
            try:
                await self._drain(cleanup)
            finally:
                # Failure to acknowledge teardown retains admission. Recovery
                # requires the worker inventory/reaper, not optimistic release.
                if cleanup.done() and not cleanup.cancelled() and cleanup.exception() is None:
                    remaining = self.active_jobs[lease.owner_id] - 1
                    if remaining:
                        self.active_jobs[lease.owner_id] = remaining
                    else:
                        del self.active_jobs[lease.owner_id]
