"""Production composition for the invited app, with explicit private configuration.

Only prepared identities, fixed-owner credentials, budgets and sealed indexes are
accepted. This launcher never provisions accounts or opens an administrator DSN.
"""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
import fcntl
import json
import logging
import os
import stat
import time

import httpx

from .auth.gmail_consent import GmailConsent
from .auth.identity_store import IdentityStore
from .auth.invited_broker import BoundGmailBroker
from .auth.public import validate_public_auth_config
from .gateway.artifacts import ArtifactStore
from .gateway.attachment_raw_service import RunRawAttachmentService
from .gateway.attachment_read_service import RunAttachmentReadService
from .gateway.attachment_reader import OwnerAttachmentReader
from .gateway.attachment_source import OwnerAttachmentSource, QueryAttachmentLocator
from .gateway.browser_conversations import compose_browser_runs, validate_receipt_schema
from .gateway.browser_mail import BrowserMail
from .gateway.capabilities import Capabilities
from .gateway.data_admission import DataAdmission
from .gateway.database import QueryGateway, QueryLimits, ReaderCredential, ReaderRegistry
from .gateway.events import Events
from .gateway.facts_service import FactsOwnerContext, RunFactsService
from .gateway.full_agent_remote import FULL_LIMITS, GUEST_PROFILES, SSHFullAgentBackend, SSHTransport, validate_bootstrap
from .gateway.http import create_gateway_app
from .gateway.maintenance import GateReader, ReleaseIdentity
from .gateway.metadata_service import RunMetadataService
from .gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT
from .gateway.provider import AnthropicRunService, ProviderProfile, _finish
from .gateway.effort_router import EffortRouter
from .gateway.escalation import LADDER_MODELS, RunEscalation
from .gateway.jev import jev_config_from_env
from .gateway.judge_http import RunJudgeService
from .gateway.gemini import MODEL as GEMINI_MODEL, GeminiProfile, GeminiRunService
from .gateway.gemini_http import GeminiHTTPTransport
from .gateway.openrouter import OpenRouterProfile, OpenRouterRunService
from .gateway.openrouter_http import OpenRouterHTTPTransport
from .gateway.provider_http import AnthropicHTTPTransport, ClaudeLoginFile
from .gateway.provision_writer import verify_writer_access
from .gateway.registry import AccessDenied, Registry
from .gateway.retrieval import RunRetrievalService
from .gateway.search_embedding import GeminiEmbeddingHTTPTransport, GeminiEmbeddingProfile, GeminiQueryEmbedder
from .gateway.search_index import IndexBinding, OwnerIndexRegistry, load_scann_index
from .gateway.search_reader import SearchCredential, SearchReader, SearchRegistry
from .gateway.search_reranker import GeminiRerankerHTTPTransport, GeminiRerankerProfile, GeminiThreadReranker
from .gateway.search_service import OwnerSearchContext, RunSearchService
from .gateway.service import RunQueryService
from .gateway.tool_deadline import TOOL_DEADLINE_SECONDS
from .gateway.worker import WorkerController
from .gateway.writer import WriterCredential, WriterRegistry, application_writer_role
from .invited_app import create_invited_app
from .invited_config import invited_search_profile

logger=logging.getLogger(__name__)


class ProductionBackend(SSHFullAgentBackend):
    namespace='production-full-agent-rpc'


@dataclass(frozen=True)
class Runtime:
    browser_app: object
    gateway_app: object


@contextmanager
def runtime_lock(state_dir):
    """A second API process must never reconcile this process's live workers."""
    fd=os.open(state_dir/'runtime.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise RuntimeError('Invalid runtime lock')
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An invited runtime already owns this state') from None
        yield
    finally:
        os.close(fd)


def require_release(config):
    identity=ReleaseIdentity(config.release.store_id,TEXT,config.release.release_epoch)
    gate=GateReader(config.state_dir/'registry.sqlite',identity)
    gate.require_ready()
    return gate


def verify_budgets(registry,owners):
    with registry._transaction() as db:
        for owner in owners:
            row=db.execute('SELECT owner_id,ceiling FROM budgets WHERE budget_id=?',(owner.budget_id,)).fetchone()
            if row is None or row['owner_id']!=owner.id or row['ceiling']<=0:
                raise AccessDenied()
            usage=db.execute('SELECT spent,reserved FROM budgets WHERE budget_id=?',(owner.budget_id,)).fetchone()
            logger.warning('token budget owner=%s ceiling=%d spent=%d reserved=%d',
                owner.id,row['ceiling'],usage['spent'],usage['reserved'])


def run_gate(owners,pending,identities):
    """Who may run: a configured, active owner whose databases startup checked.

    A pending owner may sign in (binding their subject) but runs nothing until a
    restart has run the database checks startup skipped for them.
    """
    def can_run(owner):
        return owner in owners and owner not in pending and identities.is_active(owner)
    return can_run


# How long a positive owner/release-gate answer is reused. The full check opens
# three SQLite files; readers consult it continuously, so an uncached gate made
# a 10k-row search take ~30 s. A refusal is never cached.
OWNER_GATE_CACHE_SECONDS=1.0


def recently_confirmed(check,*,ttl=OWNER_GATE_CACHE_SECONDS,clock=time.monotonic):
    """Wrap an owner check so a True answer is reused for `ttl` seconds."""
    confirmed_until={}
    def cached(owner):
        if confirmed_until.get(owner,float('-inf'))>clock():
            return True
        result=check(owner)
        if result is True:
            confirmed_until[owner]=clock()+ttl
        return result
    return cached


def _subject_matches(owner,subject):
    """A pinned subject must match; an unpinned one defers to the identity store,
    which binds the first verified sign-in and refuses any other afterwards."""
    return owner.google_subject is None or subject==owner.google_subject


def verify_identities(identities,owners):
    """Refuse any mismatch; return the owners still awaiting their first sign-in.

    Pending means unpinned here and unbound in the store. The store reports such
    an owner inactive, so nothing can run for them and their logins cannot be
    exercised yet; startup skips only their database checks. Restart after the
    first sign-in to check them in full, then pin the subject.
    """
    pending=set()
    with identities._transaction() as db:
        for owner in owners:
            row=db.execute('SELECT * FROM identities WHERE owner_id=?',(owner.id,)).fetchone()
            if (not identities._ready(db,row) or row['email']!=owner.email
                    or not _subject_matches(owner,row['google_subject'])):
                raise AccessDenied()
            if row['google_subject'] is None:
                pending.add(owner.id)
    return frozenset(pending)


# Agent VMs the worker runs at once (1 vCPU, up to ~1 GB each; the worker has
# 4 vCPUs and ~3.9 GB, and Firecracker only backs memory a guest touches).
MAX_CONCURRENT_RUNS=5
# Concurrent mail queries / embedding calls. Admission refuses (503) rather than
# waits, and a run's parent plus three subagent children overran 4 global / 2
# per owner: a quarter of their calls failed (2026-09-24).
DATA_GLOBAL_CONCURRENCY=16
DATA_OWNER_CONCURRENCY=8
# Each guest runtime speaks to the gateway through its own qualified CLI profile.
CLIENT_PROFILES={'pi':'pi-0.84.4','claude':'claude-2.1.272'}
# Guests are told this cap; the gateway refuses anything above it.
OUTPUT_TOKEN_LIMIT=4096
# Pi on Gemini. MEDIUM is the legacy Pi default; thought tokens bill as output.
GEMINI_OUTPUT_TOKEN_LIMIT=16384
GEMINI_THINKING='MEDIUM'
# Pi on OpenRouter-served models: runtime -> model. Medium effort, as legacy Pi.
OPENROUTER_RUNTIMES={'pi_opus':'anthropic/claude-opus-5'}
OPENROUTER_OUTPUT_TOKEN_LIMIT=16384


def anthropic_transports(provider):
    """One upstream credential per client profile; a runtime without one is refused.

    Claude Code prefers the subscription login and falls back to the API key.
    Pi never uses the login: Anthropic accepts it only from Claude Code.
    """
    key=AnthropicHTTPTransport(provider.anthropic_key) if provider.anthropic_key else None
    login_source=provider.claude_oauth_token or (
        ClaudeLoginFile(provider.claude_login_file) if provider.claude_login_file else None)
    login=AnthropicHTTPTransport(login_source,kind='oauth') if login_source else None
    routes={CLIENT_PROFILES['pi']:key,CLIENT_PROFILES['claude']:login or key}
    return {client:transport for client,transport in routes.items() if transport is not None}


def _bind_with(service,profile):
    """Bind this exact profile. Never close over a loop variable here: a later
    profile would silently rebind an earlier runtime's service (it did)."""
    def bind(run_id,prompt=None):
        service.bind_profile(run_id,profile)
        return None  # No planning hint.
    return bind


def _bind_routed_gemini(service,profile,router):
    """Bind Gemini on the model and thinking level Jev picks from this run's
    question; return Jev's planning hint (parallel subagents) or None."""
    def bind(run_id,prompt=None):
        if router is None or not prompt:
            service.bind_profile(run_id,profile)
            return None
        model,level,hint=router.start(prompt)
        service.bind_profile(run_id,replace(profile,model=model,thinking_level=level))
        if hint and service.escalation is not None:
            service.escalation.hold(run_id)
        return hint
    return bind


def gemini_transports_by_model(key,default,stack):
    """One upstream transport per escalation-ladder model (each checks its own
    modelVersion); the already-open default transport serves its own model."""
    transports={GEMINI_MODEL:default}
    for model in LADDER_MODELS:
        if model not in transports:
            transports[model]=GeminiHTTPTransport(key,model=model)
            stack.push_async_callback(_close_async,transports[model])
    return transports


def effort_router_for(jev):
    """Jev effort router when Jev is configured, else None (fixed MEDIUM)."""
    if jev is None:
        return None
    return EffortRouter(jev,httpx.Client(trust_env=False,follow_redirects=False))


def judge_service_for(caps,jev,escalation):
    """Guest `judge` tool backend when Jev is configured, else None (route absent)."""
    if jev is None:
        return None
    return RunJudgeService(caps,jev,httpx.AsyncClient(trust_env=False,follow_redirects=False),
        escalation=escalation)


def _profile_binders(provider,gemini,rates,openrouter=None,router=None):
    """runtime -> bind(run_id), for exactly the runtimes this deployment can serve."""
    routed=getattr(provider,'transports_by_client',None)
    binders={}
    for runtime,client in CLIENT_PROFILES.items():
        if routed is not None and client not in routed:
            continue
        profile=ProviderProfile(model='claude-sonnet-4-6',input_token_limit=200000,
            output_token_limit=OUTPUT_TOKEN_LIMIT,input_units_per_token=rates.input_units_per_token,
            output_units_per_token=rates.output_units_per_token,client_profile=client,effort='high')
        binders[runtime]=_bind_with(provider,profile)
    if gemini is not None:
        profile=GeminiProfile(GEMINI_MODEL,200000,GEMINI_OUTPUT_TOKEN_LIMIT,rates.input_units_per_token,
            rates.output_units_per_token,thinking_level=GEMINI_THINKING)
        binders['pi_gemini']=_bind_routed_gemini(gemini,profile,router)
    if openrouter is not None:
        for runtime,model in OPENROUTER_RUNTIMES.items():
            profile=OpenRouterProfile(model,200000,OPENROUTER_OUTPUT_TOKEN_LIMIT,rates.input_units_per_token,
                rates.output_units_per_token,reasoning_effort='medium')
            binders[runtime]=_bind_with(openrouter,profile)
    return binders


MAX_PROMPT_BYTES=16384


def with_planning_hint(prompt,hint):
    """Append the router's hint when it fits the guest's prompt bound."""
    if not hint:
        return prompt
    combined=prompt+'\n\n'+hint
    return combined if len(combined.encode('utf-8'))<=MAX_PROMPT_BYTES else prompt


def envelope_factory(capabilities,provider,rates,*,gemini=None,openrouter=None,router=None):
    binders=_profile_binders(provider,gemini,rates,openrouter,router)
    def envelope(lease,prompt,runtime='pi'):
        # Refuse before launch, not on the guest's first inference call.
        if runtime not in binders:
            raise AccessDenied()
        hint=binders[runtime](lease.run_id,prompt)
        prompt=with_planning_hint(prompt,hint)
        tokens={audience:capabilities.issue(lease.run_id,audience=audience,operations=operations,
            ttl=FULL_LIMITS.wall_seconds).secret
            for audience,operations in (
                ('sql',{'schema','query'}),
                ('retrieval',{'thread.get','search','facts.find','query.emails','judge'}),
                ('attachment',{'meta','text','raw'}),
                ('artifact',{'artifact.commit','artifact.read'}),
                ('inference',{'generate'}),('events',{'append'}))}
        value={'version':1,'profile':GUEST_PROFILES[runtime],'prompt':prompt,
            'tool_config':{'version':3,'tool_profile':'mail-raw-mcp-v3',
                'capabilities':{key:tokens[key] for key in ('sql','retrieval','artifact','attachment')}},
            'inference_capability':tokens['inference'],'events_capability':tokens['events']}
        raw=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
        packet=len(raw).to_bytes(4,'big')+raw
        validate_bootstrap(packet,prompt,runtime)
        return packet
    envelope.runtimes=frozenset(binders)
    return envelope


async def _close_async(resource):
    await _finish(resource.aclose())


async def _close_sync(resource):
    await _finish(asyncio.to_thread(resource.close))


async def _publish_index(indexes,binding,path):
    task=asyncio.create_task(asyncio.to_thread(load_scann_index,binding,path))
    try:
        resource=await _finish(task)
    except BaseException:
        if task.done() and not task.cancelled() and task.exception() is None:
            await _close_sync(task.result().searcher)
        raise
    try:
        await indexes.publish(resource)
    except BaseException:
        await _close_sync(resource.searcher)
        raise


async def verify_databases(owners,writers,gateway,search_reader):
    """Open fixed logins without reading message contents or changing schema."""
    def writer_check(owner):
        with writers.connection(owner.id) as db:
            validate_receipt_schema(db)
            verify_writer_access(db,application_writer_role(owner.id),owner.id)
            db.execute('SELECT run_id FROM public.browser_answer_receipts LIMIT 0')
    for owner in owners:
        await _finish(asyncio.to_thread(writer_check,owner))
        await gateway.query(owner.id,'SELECT id FROM messages WHERE 1 = 0 LIMIT 1')
        async def active():
            search_reader.registry.credential(owner.id)
            return True
        async with search_reader.session(owner.id,deadline=asyncio.get_running_loop().time()+10,check_active=active):
            pass


@asynccontextmanager
async def open_runtime(config):
    """Acquire the complete runtime and release it after both listeners drain."""
    gate=require_release(config)  # Missing/closed state creates no replacement store.
    auth=validate_public_auth_config()
    identity_secrets={auth.secret,os.environ.get('GMS_SESSION_SECRET','')} if auth else set()
    if auth is None or any(secret in identity_secrets for secret in (config.broker.bearer,config.broker.signing_secret)):
        raise AccessDenied()
    async with AsyncExitStack() as stack:
        stack.enter_context(runtime_lock(config.state_dir))
        gate.require_ready()
        # Identity creation belongs to the offline provisioning step.
        identity_path=config.state_dir/'identities.sqlite'
        info=identity_path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise AccessDenied()
        identities=IdentityStore(identity_path)
        pending=verify_identities(identities,config.owners)
        owners={owner.id:owner for owner in config.owners}
        can_run=run_gate(owners,pending,identities)
        def check_owner_and_gate(owner):
            gate.require_ready()
            return can_run(owner)
        is_active=recently_confirmed(check_owner_and_gate)
        # The registry checks the gate itself, on the transaction's own
        # connection. Its owner check must not open a second one: inside a
        # read-only capability check that deadlocked against a committing
        # writer ("database is locked") until the busy timeout (2026-09-24).
        registry=Registry(config.state_dir/'registry.sqlite',is_active=recently_confirmed(can_run),
            release_identity=gate.identity)
        verify_budgets(registry,config.owners)
        caps=Capabilities(registry)
        events=Events(caps)
        artifacts=ArtifactStore(config.state_dir/'artifacts',caps)
        readers=ReaderRegistry({o.id:ReaderCredential(o.id,o.reader_dsn) for o in config.owners},is_active=is_active)
        # Writes check the gate fresh: maintenance must stop them at once. Reads may
        # lag by the cache window; the registry re-reads the gate itself anyway.
        writers=WriterRegistry({o.id:WriterCredential(o.id,o.writer_dsn) for o in config.owners},
            is_active=check_owner_and_gate)
        admission=DataAdmission(global_concurrency=DATA_GLOBAL_CONCURRENCY,owner_concurrency=DATA_OWNER_CONCURRENCY)
        gateway=QueryGateway(readers,limits=QueryLimits(global_concurrency=DATA_GLOBAL_CONCURRENCY,
            owner_concurrency=DATA_OWNER_CONCURRENCY,
            deadline_seconds=TOOL_DEADLINE_SECONDS),admission=admission)
        search_reader=SearchReader(SearchRegistry({o.id:SearchCredential(o.id,o.search_dsn,schema_profile=TEXT)
            for o in config.owners},is_active=is_active),
            profile=invited_search_profile(config.provider.fact_model_tag),admission=admission)
        await verify_databases([o for o in config.owners if o.id not in pending],writers,gateway,search_reader)
        indexes=OwnerIndexRegistry()
        stack.push_async_callback(_close_async,indexes)
        for owner in config.owners:
            binding=IndexBinding(owner.id,owner.index.generation,'gemini-embedding-2',3072,owner.index.source_id)
            await _publish_index(indexes,binding,owner.index.path)
        broker_http=httpx.Client(trust_env=False,follow_redirects=False)
        stack.callback(broker_http.close)
        broker=BoundGmailBroker(config.broker.origin,bearer=config.broker.bearer,
            signing_secret=config.broker.signing_secret,client=broker_http)
        consent=GmailConsent(identities)
        inference_transports=anthropic_transports(config.provider)
        for transport in {id(t):t for t in inference_transports.values()}.values():
            stack.push_async_callback(_close_async,transport)
        embedding_transport=GeminiEmbeddingHTTPTransport(config.provider.gemini_key)
        stack.push_async_callback(_close_async,embedding_transport)
        rerank_transport=GeminiRerankerHTTPTransport(config.provider.gemini_key)
        stack.push_async_callback(_close_async,rerank_transport)
        provider_admission=DataAdmission(global_concurrency=DATA_GLOBAL_CONCURRENCY,
            owner_concurrency=DATA_OWNER_CONCURRENCY)
        embedder=GeminiQueryEmbedder(registry,embedding_transport,
            profile=GeminiEmbeddingProfile(config.provider.embedding_units_per_token),admission=provider_admission)
        reranker=GeminiThreadReranker(registry,rerank_transport,
            profile=GeminiRerankerProfile(config.provider.rerank_input_units_per_token,
                config.provider.rerank_output_units_per_token,'full-model-ceilings-v1'),admission=provider_admission)
        provider=AnthropicRunService(caps,None,transports_by_client=inference_transports)
        gemini_transport=GeminiHTTPTransport(config.provider.gemini_key,model=GEMINI_MODEL)
        stack.push_async_callback(_close_async,gemini_transport)
        escalation=RunEscalation()
        gemini=GeminiRunService(caps,gemini_transport,escalation=escalation,
            transports_by_model=gemini_transports_by_model(config.provider.gemini_key,gemini_transport,stack))
        openrouter=None
        if config.provider.openrouter_key:
            openrouter_transport=OpenRouterHTTPTransport(config.provider.openrouter_key)
            stack.push_async_callback(_close_async,openrouter_transport)
            openrouter=OpenRouterRunService(caps,openrouter_transport)
        attachment_reader=OwnerAttachmentReader(gateway)
        locator=QueryAttachmentLocator(gateway,config.attachment_root)
        source=OwnerAttachmentSource(config.attachment_root,locate=locator.locate_raw)
        jev=jev_config_from_env()
        gateway_app=create_gateway_app(RunQueryService(caps,gateway),artifacts=artifacts,
            retrieval=RunRetrievalService(caps,gateway,attachment_reader=attachment_reader),
            search=RunSearchService(caps,search_reader,indexes,embedder,
                owners={o.id:OwnerSearchContext(o.id,(o.email,)) for o in config.owners},reranker=reranker),
            facts=RunFactsService(caps,search_reader,embedder,
                owners={o.id:FactsOwnerContext(o.id,o.email) for o in config.owners}),
            metadata=RunMetadataService(caps,gateway),
            attachment_reads=RunAttachmentReadService(caps,attachment_reader),
            raw_attachments=RunRawAttachmentService(caps,source,
                admission=DataAdmission(global_concurrency=2,owner_concurrency=1)),
            anthropic=provider,gemini=gemini,openrouter=openrouter,events=events,
            judge=judge_service_for(caps,jev,escalation))
        transport=SSHTransport(host=config.worker.host,port=config.worker.port,
            private_key=config.worker.private_key,known_hosts=config.worker.known_hosts)
        envelope_for=envelope_factory(caps,provider,config.provider,gemini=gemini,openrouter=openrouter,
            router=effort_router_for(jev))
        backend=ProductionBackend(registry,transport=transport,envelope_for=envelope_for)
        stack.push_async_callback(_close_sync,backend)
        workers=WorkerController(registry,backend,limits=FULL_LIMITS,max_workers=MAX_CONCURRENT_RUNS,
            max_owner_workers=MAX_CONCURRENT_RUNS)
        runs,conversations=compose_browser_runs(workers,events,connect=None,is_active=is_active,
            prepare_input=backend.prepare_input,budget_for=lambda owner:owners[owner].budget_id,
            transaction_factory=writers.connection)
        # Also owns cleanup if app creation or either listener startup fails.
        stack.push_async_callback(runs.close)
        def provision_account(account,claims):
            owner=owners.get(account.owner_id)
            if (owner is None or account.email!=owner.email or claims.email!=owner.email
                    or not _subject_matches(owner,claims.subject) or claims.email_verified is not True
                    or not (owner.id in pending or is_active(owner.id))):
                raise AccessDenied()
            return True
        # Revoke abandoned run authority before even the gateway socket opens.
        await runs.recover()
        await _finish(asyncio.to_thread(broker.drain_cleanup,consent))
        browser_app=create_invited_app(identities=identities,consent=consent,broker=broker,
            provision_account=provision_account,runs=runs,conversations=conversations,
            artifacts=artifacts,mail=BrowserMail(gateway,attachment_reader=attachment_reader,
                attachment_source=source),startup_prepared=True,runtimes=envelope_for.runtimes)
        gate.require_ready()
        yield Runtime(browser_app,gateway_app)
