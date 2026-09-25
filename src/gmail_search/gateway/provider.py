"""Run-bound Anthropic streaming service; no public route.

The injected transport is trusted infrastructure. Its async ``stream`` context
manager receives only this service's fixed URL, normalized body and disabled
redirect flag; credentials and provider headers belong inside that adapter. It
must yield a response with status_code, bounded byte chunks and ProviderUsage
parsed from a complete trusted provider response. Guest usage is never accepted.
Pinned Pi/native Claude single-tool exchanges are qualified only against a
synthetic upstream; complete sessions and live provider behavior remain unqualified.

Profile limits/rates are trusted server configuration, not current price claims.
Input limits must cover the provider model's entire possible billable context,
including hidden framing/image tokens. Reserve that worst case before dispatch.
Missing/invalid usage or interrupted responses charge the entire reservation.
"""
import asyncio
import logging
from dataclasses import asdict, dataclass
import json
import math

from .capabilities import Capabilities
from .inference import MAX_OUTPUT_TOKENS, validate_server_effort
from .cli_compat import CLIENT_PROFILES, compile_anthropic_cli_request
from .registry import AccessDenied

_ENDPOINT = 'https://api.anthropic.com/v1/messages'

logger = logging.getLogger(__name__)
# Re-authorization interval for an open model stream. At 50 ms, four parallel
# streams made ~80 registry reads a second (2026-09-24).
WATCH_SECONDS = .25


@dataclass(frozen=True)
class ProviderProfile:
    model: str
    input_token_limit: int
    output_token_limit: int
    input_units_per_token: int
    output_units_per_token: int
    timeout_seconds: float = 60
    max_response_bytes: int = 16 * 1024 * 1024
    client_profile: str = 'strict'
    effort: str | None = None
    family: str = 'anthropic'

    def __post_init__(self):
        try:
            validate_server_effort(self.model,self.effort)
        except ValueError:
            raise AccessDenied() from None
        if (self.family != 'anthropic' or type(self.client_profile) is not str or self.client_profile not in CLIENT_PROFILES
                or type(self.model) is not str or not self.model or len(self.model) > 128
                or any(type(value) is not int or not 1 <= value <= 10**9 for value in (
                    self.input_token_limit, self.output_token_limit, self.input_units_per_token,
                    self.output_units_per_token, self.max_response_bytes))
                or self.output_token_limit > MAX_OUTPUT_TOKENS
                or self.max_response_bytes > 64 * 1024 * 1024
                or type(self.timeout_seconds) not in (int,float)
                or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 3600
                or self.input_token_limit*self.input_units_per_token + self.output_token_limit*self.output_units_per_token > 10**12):
            raise AccessDenied()


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int


class ReplayRejected(AccessDenied):
    """A prior claim already owns this request; never repeat its billable call.

    Results are not persisted here. Retry/reconnect support requires a separate
    durable event spool; callers must not fabricate new request keys to retry.
    """


async def _finish(awaitable):
    """Drain owned work before propagating any repeated caller cancellation."""
    task = asyncio.ensure_future(awaitable)
    interrupted = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError()
    return result


class AnthropicRunService:
    profile_type = ProviderProfile

    def _endpoint(self, profile):
        return _ENDPOINT

    def _compile(self, body, profile):
        normalized = compile_anthropic_cli_request(body,server_model=profile.model,max_output_tokens=profile.output_token_limit,client_profile=profile.client_profile,server_effort=profile.effort)
        if normalized.get('stream') is not True:
            raise AccessDenied()
        return normalized

    def _output_limit(self, body):
        return body['max_tokens']

    def __init__(self, capabilities: Capabilities, transport, *, transports_by_client=None):
        """`transports_by_client` routes each client profile to its own upstream
        credential; a profile with no entry is refused rather than defaulted."""
        self.capabilities, self.transport = capabilities, transport
        self.transports_by_client = transports_by_client
        self.registry = capabilities.registry
        with self.registry._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS provider_profiles (run_id TEXT PRIMARY KEY, profile TEXT NOT NULL)')

    def bind_profile(self, run_id, profile: ProviderProfile):
        """Trusted controller only; immutable durable binding before issuing access."""
        if type(profile) is not self.profile_type:
            raise AccessDenied()
        encoded = json.dumps(asdict(profile), sort_keys=True)
        with self.registry._transaction() as db:
            self.registry._active(db, run_id)
            old = db.execute('SELECT profile FROM provider_profiles WHERE run_id=?', (run_id,)).fetchone()
            if old and old['profile'] != encoded:
                raise AccessDenied()
            db.execute('INSERT OR IGNORE INTO provider_profiles VALUES(?,?)', (run_id, encoded))

    def profile(self, run_id):
        with self.registry._transaction() as db:
            self.registry._active(db, run_id)
            row = db.execute('SELECT profile FROM provider_profiles WHERE run_id=?', (run_id,)).fetchone()
            if row is None:
                raise AccessDenied()
            try:
                return self.profile_type(**json.loads(row['profile']))
            except (TypeError, ValueError):
                raise AccessDenied() from None

    async def _authorize(self, token):
        return await asyncio.to_thread(self.capabilities.authorize, token, audience='inference', operation='generate')

    async def _watch(self, token):
        # Every chunk is also authorized; this bounds a stalled stream.
        while True:
            await asyncio.sleep(WATCH_SECONDS)
            await self._authorize(token)

    @staticmethod
    def _charge(usage, profile, output_limit, reservation):
        if (type(usage) is not ProviderUsage
                or type(usage.input_tokens) is not int or type(usage.output_tokens) is not int
                or not 0 <= usage.input_tokens <= profile.input_token_limit
                or not 0 <= usage.output_tokens <= output_limit):
            return reservation
        return usage.input_tokens*profile.input_units_per_token + usage.output_tokens*profile.output_units_per_token

    async def _produce(self, token, body, profile, queue):
        async with self._transport_for(profile).stream(url=self._endpoint(profile), body=body, follow_redirects=False) as response:
            if response.status_code != 200:
                logger.warning('upstream %s returned %s', profile.model, response.status_code)
                raise AccessDenied()
            total = 0
            async for chunk in response:
                if type(chunk) is not bytes or len(chunk) > 1024*1024:
                    logger.warning('upstream %s sent an invalid chunk', profile.model)
                    raise AccessDenied()
                total += len(chunk)
                if total > profile.max_response_bytes:
                    logger.warning('upstream %s response over %d bytes', profile.model, profile.max_response_bytes)
                    raise AccessDenied()
                await self._authorize(token)
                await queue.put(chunk)
            await self._authorize(token)
            return response.usage

    def _transport_for(self, profile):
        if self.transports_by_client is None:
            return self.transport
        transport = self.transports_by_client.get(profile.client_profile)
        if transport is None:
            raise AccessDenied()
        return transport

    async def _drive(self, token, lease, request_key, body, profile, units, queue):
        producer = watch = None
        charge = units
        try:
            # Recheck after reservation: a concurrent revocation must not initiate
            # transport. The durable claim remains consumed even on this failure.
            await self._authorize(token)
            duration = min(profile.timeout_seconds, lease.deadline-self.registry.clock())
            async with asyncio.timeout(max(0,duration)):
                producer = asyncio.create_task(self._produce(token,body,profile,queue))
                watch = asyncio.create_task(self._watch(token))
                done, _ = await asyncio.wait((producer,watch), return_when=asyncio.FIRST_COMPLETED)
                if watch in done:
                    await watch
                usage = await producer
                await self._authorize(token)
                charge = self._charge(usage,profile,self._output_limit(body),units)
        except (AccessDenied, TimeoutError):
            raise
        except Exception:
            raise RuntimeError('Provider transport failed.') from None
        finally:
            async def cleanup():
                pending = [task for task in (producer,watch) if task is not None]
                for task in pending:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                # Settlement intentionally works after token/run revocation.
                await asyncio.to_thread(self.registry.settle,lease.run_id,request_key,charge)
            await _finish(cleanup())

    def _effective_profile(self, lease, profile):
        """The bound profile, as this call should use it (providers may escalate)."""
        return profile

    async def stream(self, token, request_key, body):
        """Yield bounded provider bytes with authorization before every publication."""
        lease = await self._authorize(token)
        profile = self._effective_profile(lease, await asyncio.to_thread(self.profile,lease.run_id))
        normalized = self._compile(body, profile)
        units = profile.input_token_limit*profile.input_units_per_token + self._output_limit(normalized)*profile.output_units_per_token
        reservation = asyncio.create_task(asyncio.to_thread(
            self.registry.reserve,lease.run_id,request_key,units))
        try:
            claim = await asyncio.shield(reservation)
        except asyncio.CancelledError:
            async def abandon_claim():
                claim = await reservation
                # The thread can commit after cancellation. Only this caller's
                # new claim may be settled; a replay belongs to another caller.
                if claim.created:
                    await asyncio.to_thread(self.registry.settle,lease.run_id,request_key,units)
            await _finish(abandon_claim())
            raise
        if not claim.created:
            raise ReplayRejected()
        queue = asyncio.Queue(maxsize=1)
        driver = asyncio.create_task(self._drive(token,lease,request_key,normalized,profile,units,queue))
        getter = None
        try:
            while True:
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait((getter,driver),return_when=asyncio.FIRST_COMPLETED)
                if driver in done:
                    await driver  # Raise before publishing buffered data on failure.
                    if not getter.done() and queue.empty():
                        break
                chunk = await getter
                getter = None
                await self._authorize(token)
                yield chunk
                if driver.done() and queue.empty():
                    await driver
                    break
        finally:
            async def cleanup():
                if getter is not None:
                    getter.cancel()
                if not driver.done() and not driver.cancelling():
                    driver.cancel()
                await asyncio.gather(driver,*([getter] if getter is not None else []),return_exceptions=True)
            await _finish(cleanup())
