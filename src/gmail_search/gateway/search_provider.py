"""Private shared lifecycle for fixed internal search providers.

Concrete adapters own immutable model/body/usage profiles. Shared code owns run
binding, budget claims, provider capacity and cancellation-resistant teardown.
No guest route or endpoint selection lives here.
"""
import asyncio
from contextlib import asynccontextmanager
import math
import uuid

from .data_admission import DataAdmission
from .provider import _finish
from .registry import AccessDenied, RunLease


@asynccontextmanager
async def _owned_stream(context):
    """Own exit separately so cancellation cannot abandon a normal EOF close."""
    response=await context.__aenter__()
    try:
        yield response
    except BaseException as error:
        await _finish(context.__aexit__(type(error),error,error.__traceback__))
        raise
    else:
        await _finish(context.__aexit__(None,None,None))


class _SearchProvider:
    def __init__(self, registry, transport, *, profile, admission):
        if (type(profile) is not self.profile_type or type(admission) is not DataAdmission
                or admission.global_concurrency>4 or admission.owner_concurrency>2):
            raise ValueError('Invalid search provider configuration')
        self._registry=registry
        self._transport=transport
        self._profile=profile
        self._admission=admission
        self._quarantine={}
        self._loop=None

    @property
    def model(self):
        return self._profile.model

    def _event_loop(self):
        loop=asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise self.error_type()
        self._loop=loop
        return loop

    def _binding(self, lease):
        if type(lease) is not RunLease:
            raise AccessDenied()
        with self._registry._transaction() as db:
            row=self._registry._active(db,lease.run_id)
            current=self._registry._lease(row)
            if any(getattr(current,key)!=getattr(lease,key) for key in RunLease.__dataclass_fields__ if key!='lease_expires'):
                raise AccessDenied()
            self._registry._fence(db,row)

    async def _check(self, lease, check_active, deadline):
        if asyncio.get_running_loop().time()>=deadline:
            raise TimeoutError()
        await _finish(asyncio.to_thread(self._binding,lease))
        if await check_active() is not True:
            raise AccessDenied()
        if asyncio.get_running_loop().time()>=deadline:
            raise TimeoutError()

    async def _produce(self, lease, body, check_active, deadline):
        chunks=[]; size=0
        context=self._transport.stream(url=self.endpoint,body=body,follow_redirects=False)
        async with _owned_stream(context) as response:
            if type(response.status_code) is not int or response.status_code!=200:
                raise self.error_type()
            async for chunk in response:
                if type(chunk) is not bytes or len(chunk)>65536:
                    raise self.error_type()
                size+=len(chunk)
                if size>self.max_response_bytes:
                    raise self.error_type()
                await self._check(lease,check_active,deadline)
                chunks.append(chunk)
            parsed=self._parse_response(b''.join(chunks),body)
        await self._check(lease,check_active,deadline)
        return parsed

    async def _operation(self, lease, body, check_active, deadline):
        await self._check(lease,check_active,deadline)
        try:
            capacity=self._admission.acquire(lease.owner_id)
        except RuntimeError:
            raise self.error_type() from None
        key=self.request_prefix+uuid.uuid4().hex
        units=self._reservation_units()
        charge=units
        reservation=producer=watch=None
        try:
            reservation=asyncio.create_task(asyncio.to_thread(self._registry.reserve,lease.run_id,key,units))
            claim=await asyncio.shield(reservation)
            if not claim.created:
                raise self.error_type()
            await self._check(lease,check_active,deadline)
            async def watching():
                while True:
                    await asyncio.sleep(.05)
                    await self._check(lease,check_active,deadline)
            async with asyncio.timeout_at(deadline):
                producer=asyncio.create_task(self._produce(lease,body,check_active,deadline))
                watch=asyncio.create_task(watching())
                done,_=await asyncio.wait((producer,watch),return_when=asyncio.FIRST_COMPLETED)
                if watch in done:
                    await watch
                result,usage=await producer
                await self._check(lease,check_active,deadline)
                charge=self._charge(usage)
                return result
        finally:
            async def cleanup():
                tasks=[task for task in (producer,watch) if task is not None]
                for task in tasks:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                claim=None
                if reservation is not None:
                    try:
                        claim=await reservation
                    except Exception:
                        pass  # Registry reservation failure rolls its transaction back.
                if claim is not None and claim.created:
                    try:
                        await asyncio.to_thread(self._registry.settle,lease.run_id,key,charge)
                    except Exception:
                        self._quarantine[key]=(lease.run_id,charge,capacity)
                        raise self.error_type() from None
                capacity.release()
            await _finish(cleanup())

    async def _call(self, lease, body, *, deadline, check_active):
        loop=self._event_loop()
        if type(deadline) not in (int,float) or not math.isfinite(deadline):
            raise ValueError('Invalid provider deadline')
        deadline=min(deadline,loop.time()+30)
        task=asyncio.create_task(self._operation(lease,body,check_active,deadline))
        try:
            try:
                result=await asyncio.shield(task)
            finally:
                if not task.done() and not task.cancelling():
                    task.cancel()
                await _finish(asyncio.gather(task,return_exceptions=True))
            # Drain all owned tasks before the final binding/auth/deadline
            # checks. No further suspension follows fresh authorization.
            await self._check(lease,check_active,deadline)
            return result
        except (AccessDenied,TimeoutError,asyncio.CancelledError):
            raise
        except Exception:
            raise self.error_type() from None

    async def reconcile(self):
        """Trusted explicit retry of failed settlement only; never repeat a call."""
        self._event_loop()
        async def finish():
            for key,(run_id,charge,capacity) in list(self._quarantine.items()):
                try:
                    await asyncio.to_thread(self._registry.settle,run_id,key,charge)
                except Exception:
                    raise self.error_type() from None
                capacity.release()
                self._quarantine.pop(key,None)
        await _finish(finish())

