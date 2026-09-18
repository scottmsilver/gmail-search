"""Production composition for the invited app, with explicit private configuration.

Only prepared identities, fixed-owner credentials, budgets and sealed indexes are
accepted. This launcher never provisions accounts or opens an administrator DSN.
"""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
import stat

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
from .gateway.provider_http import AnthropicHTTPTransport, ClaudeLoginFile
from .gateway.provision_writer import verify_writer_access
from .gateway.registry import AccessDenied, Registry
from .gateway.retrieval import RunRetrievalService
from .gateway.search_embedding import GeminiEmbeddingHTTPTransport, GeminiEmbeddingProfile, GeminiQueryEmbedder
from .gateway.search_index import IndexBinding, OwnerIndexRegistry, load_scann_index
from .gateway.search_reader import SearchCredential, SearchProfile, SearchReader, SearchRegistry
from .gateway.search_reranker import GeminiRerankerHTTPTransport, GeminiRerankerProfile, GeminiThreadReranker
from .gateway.search_service import OwnerSearchContext, RunSearchService
from .gateway.service import RunQueryService
from .gateway.worker import WorkerController
from .gateway.writer import WriterCredential, WriterRegistry, application_writer_role
from .invited_app import create_invited_app


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


def verify_identities(identities,owners):
    with identities._transaction() as db:
        for owner in owners:
            row=db.execute('SELECT * FROM identities WHERE owner_id=?',(owner.id,)).fetchone()
            if (not identities._ready(db,row) or row['email']!=owner.email
                    or row['google_subject']!=owner.google_subject):
                raise AccessDenied()


# Each guest runtime speaks to the gateway through its own qualified CLI profile.
CLIENT_PROFILES={'pi':'pi-0.84.4','claude':'claude-2.1.272'}
# Guests are told this cap; the gateway refuses anything above it.
OUTPUT_TOKEN_LIMIT=4096


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


def envelope_factory(capabilities,provider,rates):
    profiles={runtime:ProviderProfile(model='claude-sonnet-4-6',input_token_limit=200000,
        output_token_limit=OUTPUT_TOKEN_LIMIT,input_units_per_token=rates.input_units_per_token,
        output_units_per_token=rates.output_units_per_token,client_profile=client,effort='high')
        for runtime,client in CLIENT_PROFILES.items()}
    routed=getattr(provider,'transports_by_client',None)
    def envelope(lease,prompt,runtime='pi'):
        # Refuse before launch, not on the guest's first inference call.
        if runtime not in profiles or (routed is not None and CLIENT_PROFILES[runtime] not in routed):
            raise AccessDenied()
        provider.bind_profile(lease.run_id,profiles[runtime])
        tokens={audience:capabilities.issue(lease.run_id,audience=audience,operations=operations,ttl=180).secret
            for audience,operations in (
                ('sql',{'schema','query'}),
                ('retrieval',{'thread.get','search','facts.find','query.emails'}),
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
        verify_identities(identities,config.owners)
        owners={owner.id:owner for owner in config.owners}
        def is_active(owner):
            gate.require_ready()
            return owner in owners and identities.is_active(owner)
        registry=Registry(config.state_dir/'registry.sqlite',is_active=is_active,release_identity=gate.identity)
        verify_budgets(registry,config.owners)
        caps=Capabilities(registry)
        events=Events(caps)
        artifacts=ArtifactStore(config.state_dir/'artifacts',caps)
        readers=ReaderRegistry({o.id:ReaderCredential(o.id,o.reader_dsn) for o in config.owners},is_active=is_active)
        writers=WriterRegistry({o.id:WriterCredential(o.id,o.writer_dsn) for o in config.owners},is_active=is_active)
        admission=DataAdmission(global_concurrency=4,owner_concurrency=2)
        gateway=QueryGateway(readers,limits=QueryLimits(global_concurrency=4,owner_concurrency=2),admission=admission)
        search_reader=SearchReader(SearchRegistry({o.id:SearchCredential(o.id,o.search_dsn,schema_profile=TEXT)
            for o in config.owners},is_active=is_active),
            profile=SearchProfile('gemini-embedding-2',config.provider.fact_model_tag,3072,schema_profile=TEXT),admission=admission)
        await verify_databases(config.owners,writers,gateway,search_reader)
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
        provider_admission=DataAdmission(global_concurrency=4,owner_concurrency=2)
        embedder=GeminiQueryEmbedder(registry,embedding_transport,
            profile=GeminiEmbeddingProfile(config.provider.embedding_units_per_token),admission=provider_admission)
        reranker=GeminiThreadReranker(registry,rerank_transport,
            profile=GeminiRerankerProfile(config.provider.rerank_input_units_per_token,
                config.provider.rerank_output_units_per_token,'full-model-ceilings-v1'),admission=provider_admission)
        provider=AnthropicRunService(caps,None,transports_by_client=inference_transports)
        attachment_reader=OwnerAttachmentReader(gateway)
        locator=QueryAttachmentLocator(gateway,config.attachment_root)
        source=OwnerAttachmentSource(config.attachment_root,locate=locator.locate_raw)
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
            anthropic=provider,events=events)
        transport=SSHTransport(host=config.worker.host,port=config.worker.port,
            private_key=config.worker.private_key,known_hosts=config.worker.known_hosts)
        backend=ProductionBackend(registry,transport=transport,envelope_for=envelope_factory(caps,provider,config.provider))
        stack.push_async_callback(_close_sync,backend)
        workers=WorkerController(registry,backend,limits=FULL_LIMITS,max_workers=1,max_owner_workers=1)
        runs,conversations=compose_browser_runs(workers,events,connect=None,is_active=is_active,
            prepare_input=backend.prepare_input,budget_for=lambda owner:owners[owner].budget_id,
            transaction_factory=writers.connection)
        # Also owns cleanup if app creation or either listener startup fails.
        stack.push_async_callback(runs.close)
        def provision_account(account,claims):
            owner=owners.get(account.owner_id)
            if (owner is None or account.email!=owner.email or claims.email!=owner.email
                    or claims.subject!=owner.google_subject or claims.email_verified is not True
                    or not is_active(owner.id)):
                raise AccessDenied()
            return True
        # Revoke abandoned run authority before even the gateway socket opens.
        await runs.recover()
        await _finish(asyncio.to_thread(broker.drain_cleanup,consent))
        browser_app=create_invited_app(identities=identities,consent=consent,broker=broker,
            provision_account=provision_account,runs=runs,conversations=conversations,
            artifacts=artifacts,mail=BrowserMail(gateway,attachment_reader=attachment_reader,
                attachment_source=source),startup_prepared=True)
        gate.require_ready()
        yield Runtime(browser_app,gateway_app)
