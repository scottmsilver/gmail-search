"""Run-bound opaque payload ownership through an awaited transport callback.

The required admission instance is shared by all raw download handlers in this
gateway process. The source is an async, cancellation-cooperative opaque reader;
no parser, provider, filesystem writes or guest-selected owner lives here.
"""
import asyncio
import inspect
import math

from .attachment_source import RawAttachmentInput, _raw_mime
from .attachment_sandbox import INPUT_BYTES
from .data_admission import DataAdmission
from .provider import _finish
from .registry import AccessDenied, RunLease


class RawAttachmentUnavailable(RuntimeError):
    def __init__(self):
        super().__init__('Attachment bytes are unavailable.')


class RawAttachmentBusy(RuntimeError):
    def __init__(self):
        super().__init__('Attachment download capacity is unavailable.')


class _DeliveryFailed(RuntimeError):
    def __init__(self):
        super().__init__('Attachment delivery failed.')


class RunRawAttachmentService:
    def __init__(self,capabilities,raw_source,*,admission,timeout_seconds=30,watch_interval=.05):
        if (type(admission) is not DataAdmission or admission.global_concurrency>2
                or admission.owner_concurrency>1 or not inspect.iscoroutinefunction(getattr(raw_source,'load_raw',None))
                or type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds)
                or not 0<timeout_seconds<=30 or type(watch_interval) not in (int,float)
                or not math.isfinite(watch_interval) or not 0<watch_interval<=.05):
            raise ValueError('Invalid raw attachment configuration')
        self.capabilities=capabilities
        self.raw_source=raw_source
        self.admission=admission
        self.timeout_seconds=timeout_seconds
        self.watch_interval=watch_interval

    async def authorize(self,token):
        lease=await _finish(asyncio.to_thread(self.capabilities.authorize,token,
                            audience='attachment',operation='raw'))
        if type(lease) is not RunLease:
            raise AccessDenied()
        return lease

    @staticmethod
    def _source(source,lease,attachment_id):
        if (type(source) is not RawAttachmentInput or type(source.owner_id) is not str or source.owner_id!=lease.owner_id
                or type(source.attachment_id) is not int or source.attachment_id!=attachment_id
                or not _raw_mime(source.mime_type) or type(source.data) is not bytes
                or len(source.data)>INPUT_BYTES):
            raise RawAttachmentUnavailable()

    async def deliver(self,token,attachment_id,*,deadline,publish):
        """Own bytes until ``publish(source, check_active)`` and teardown finish.

        ``publish`` must await all transport sends and cooperatively acknowledge
        cancellation. It must not retain source bytes after returning/raising.
        It receives a fresh run/deadline check to call before each bounded send.
        """
        loop=asyncio.get_running_loop()
        if (type(attachment_id) is not int or not 0<attachment_id<=9223372036854775807
                or type(deadline) not in (int,float) or not math.isfinite(deadline)
                or not inspect.iscoroutinefunction(publish)):
            raise ValueError('Invalid raw attachment request')
        deadline=min(deadline,loop.time()+self.timeout_seconds)
        capacity=producer=watch=None
        try:
            async with asyncio.timeout_at(deadline):
                if loop.time()>=deadline:
                    raise TimeoutError()
                lease=await self.authorize(token)
                if loop.time()>=deadline:
                    raise TimeoutError()
                try:
                    capacity=self.admission.acquire(lease.owner_id)
                except RuntimeError:
                    raise RawAttachmentBusy() from None

                async def check():
                    if loop.time()>=deadline:
                        raise TimeoutError()
                    current=await self.authorize(token)
                    if any(getattr(current,key)!=getattr(lease,key) for key in RunLease.__dataclass_fields__
                           if key!='lease_expires'):
                        raise AccessDenied()
                    if loop.time()>=deadline:
                        raise TimeoutError()
                    return True

                async def produce():
                    source=None
                    failure=None
                    loading=True
                    try:
                        await check()
                        source=await self.raw_source.load_raw(lease.owner_id,attachment_id)
                        self._source(source,lease,attachment_id)
                        loading=False
                        await check()
                        await publish(source,check)
                    except asyncio.CancelledError:
                        failure=asyncio.CancelledError
                    except AccessDenied:
                        failure=AccessDenied
                    except TimeoutError:
                        failure=TimeoutError
                    except RawAttachmentUnavailable:
                        failure=RawAttachmentUnavailable
                    except Exception:
                        failure=RawAttachmentUnavailable if loading else _DeliveryFailed
                    finally:
                        source=None
                    # Raise outside the original except context. Loader,
                    # validator and publisher traceback frames can own raw
                    # bytes even after their coroutines finished. Propagate a
                    # fresh sanitized failure after dropping those references.
                    if failure is not None:
                        raise failure()

                async def watching():
                    while True:
                        await asyncio.sleep(self.watch_interval)
                        await check()

                producer=asyncio.create_task(produce())
                watch=asyncio.create_task(watching())
                done,_=await asyncio.wait((producer,watch),return_when=asyncio.FIRST_COMPLETED)
                if watch in done:
                    await watch
                await producer
        finally:
            async def cleanup():
                tasks=[task for task in (producer,watch) if task is not None]
                for task in tasks:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
            cleanup_task=asyncio.create_task(cleanup())
            try:
                await _finish(cleanup_task)
            finally:
                if cleanup_task.done() and not cleanup_task.cancelled() and cleanup_task.exception() is None:
                    producer=watch=None
                    if capacity is not None:
                        capacity.release()
