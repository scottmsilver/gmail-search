"""Unmounted release router exercised with real signed handoffs and a fake broker."""
import time
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import jwt
import pytest

from gmail_search.auth.identity_store import IdentityDenied, IdentityStore, VerifiedGoogleIdentity
from gmail_search.auth.gmail_consent import GmailConsent
from gmail_search.auth.invited_broker import BoundGmailBroker, BrokerUnavailable
from gmail_search.auth.invited_routes import create_invited_auth_router


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    for name, value in {
        'GMAIL_MULTI_TENANT':'1', 'GMS_PUBLIC_ORIGIN':'https://gms.example.test',
        'GMS_PUBLIC_ALLOWED_EMAILS':'original@example.test', 'GMS_IDENTITY_BROKER_URL':'https://identity.example.test',
        'GMS_IDENTITY_HANDOFF_SECRET':'i'*48, 'GMS_SESSION_SECRET':'s'*48,
    }.items():
        monkeypatch.setenv(name,value)
    store=IdentityStore(tmp_path/'identities.sqlite')
    consent=GmailConsent(store)
    provisioned=[]
    posted=[]
    broker_failure=[False]
    def transport(request):
        import json
        assert request.url.host=='gmail-broker.example.test'
        assert request.headers['authorization']=='Bearer '+'b'*48
        body=json.loads(request.content)
        posted.append((request.url.path,body))
        if broker_failure[0]:
            return raw_response(503)
        if request.url.path.endswith('/consents'):
            return raw_response(200,json={'url':'https://gmail-broker.example.test/v1/gmail/start?ticket='+'a'*64})
        if request.url.path.endswith('/disconnect'):
            return raw_response(200,json={'disconnected':True})
        raise AssertionError('Unexpected broker operation')
    http=httpx.Client(trust_env=False,transport=httpx.MockTransport(transport))
    broker=BoundGmailBroker('https://gmail-broker.example.test',bearer='b'*48,
                           signing_secret='h'*48,client=http)
    def provision(account,claims):
        provisioned.append((account,claims))
        return True
    app=FastAPI()
    app.include_router(create_invited_auth_router(store,consent,broker,provision_account=provision))
    with TestClient(app,base_url='https://gms.example.test') as client:
        yield store,consent,broker,client,provisioned,posted,broker_failure
    http.close()


def login(client,email='alice@example.test',subject='google-alice'):
    start=client.get('/api/auth/login',params={'return_url':'/mail'},follow_redirects=False)
    nonce=parse_qs(urlsplit(start.headers['location']).query)['nonce'][0]
    now=int(time.time())
    token=jwt.encode(dict(iss='https://identity.example.test',aud='gmail-search',sub=subject,email=email,
        email_verified=True,iat=now,exp=now+60,nonce=nonce,jti=nonce),'i'*48,algorithm='HS256')
    return client.get('/api/auth/callback',params={'silver_oauth':token},follow_redirects=False)


def connect(client):
    return client.post('/api/auth/connect-gmail',headers={'origin':'https://gms.example.test'},follow_redirects=False)


def finish(client,posted,**overrides):
    binding=next(body for path,body in reversed(posted) if path.endswith('/consents'))
    now=int(time.time())
    claims=dict(iss='https://gmail-broker.example.test',aud='gmail-search:gmail-consent',
        sub=binding['subject'],email=binding['email'],email_verified=True,owner_id=binding['owner_id'],
        invitation_generation=binding['invitation_generation'],credential_generation=binding['credential_generation'],
        nonce=binding['state'],jti='result-'+binding['state'],iat=now,exp=now+60)
    claims.update(overrides)
    token=jwt.encode(claims,'h'*48,algorithm='HS256')
    return client.get('/api/auth/gmail-callback',params={'gmail_consent':token},follow_redirects=False)


def test_complete_staged_browser_flow_keeps_credentials_server_side(setup):
    store,_,_,client,provisioned,posted,_=setup
    owner=store.invite('alice@example.test')
    response=login(client)
    assert response.status_code==303
    assert response.headers['location']=='/mail'
    cookie=response.headers['set-cookie']
    assert '__Host-gms_session=' in cookie and 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=lax' in cookie
    assert provisioned[0][0].owner_id==owner.owner_id
    me=client.get('/api/auth/me').json()
    assert me['user']['id']==owner.owner_id
    # The picker offers exactly what the run route serves, and never battles.
    assert me['capabilities']=={'full_runtime':False,'deep_models':{
        'pi':['google/gemini-3.8-flash'],'claude_code':['sonnet']}}
    assert client.get('/api/auth/gmail-status').json()=={'multi_tenant':True,'connect_method':'POST','connected':False}
    assert connect(client).status_code==303
    assert posted[-1][1]['owner_id']==owner.owner_id
    assert finish(client,posted).status_code==303
    assert client.get('/api/auth/gmail-status').json()=={'multi_tenant':True,'connect_method':'POST','connected':True}
    response=client.post('/api/auth/disconnect-gmail',headers={'origin':'https://gms.example.test'})
    assert response.status_code==200
    assert client.get('/api/auth/gmail-status').json()=={'multi_tenant':True,'connect_method':'POST','connected':False}
    assert client.get('/api/auth/me').json()['user']['id']==owner.owner_id
    assert posted[-1][0]=='/v1/gmail/disconnect'
    assert 'access_token' not in response.text and 'refresh_token' not in response.text


def test_unknown_or_uninvited_identity_never_reaches_provisioning(setup):
    _,_,_,client,provisioned,posted,_=setup
    assert login(client).status_code==401
    assert provisioned==[] and posted==[]


def test_wrong_google_account_never_connects_or_replaces_signed_in_user(setup):
    store,_,_,client,_,posted,_=setup
    alice=store.invite('alice@example.test')
    login(client)
    connect(client)
    assert finish(client,posted,sub='google-bob',email='bob@example.test').status_code==401
    assert client.get('/api/auth/gmail-status').json()=={'multi_tenant':True,'connect_method':'POST','connected':False}
    assert client.get('/api/auth/me').json()['user']['id']==alice.owner_id
    assert finish(client,posted).status_code==401


def test_mutations_require_exact_origin_and_forbid_caller_owner_fields(setup):
    store,_,_,client,_,posted,_=setup
    store.invite('alice@example.test');login(client)
    for path in ('/api/auth/connect-gmail','/api/auth/disconnect-gmail','/api/auth/logout'):
        assert client.post(path).status_code==403
        assert client.post(path,headers={'origin':'https://gms.example.test.evil.test'}).status_code==403
        assert client.post(path,headers={'origin':'https://gms.example.test'},json={'owner_id':'foreign'}).status_code==400
    assert posted==[]
    assert client.get('/api/auth/connect-gmail').status_code==405


def test_failed_broker_disconnect_stays_locally_disabled_and_durably_queued(setup):
    store,consent,_,client,_,posted,fail=setup
    store.invite('alice@example.test');login(client);connect(client);finish(client,posted)
    fail[0]=True
    response=client.post('/api/auth/disconnect-gmail',headers={'origin':'https://gms.example.test'})
    assert response.status_code==503
    assert client.get('/api/auth/gmail-status').json()=={'multi_tenant':True,'connect_method':'POST','connected':False}
    assert consent.pending_cleanup()


def test_logout_and_invitation_revoke_deny_subsequent_browser_requests(setup):
    store,_,_,client,_,_,_=setup
    account=store.invite('alice@example.test');login(client)
    assert client.post('/api/auth/logout',headers={'origin':'https://gms.example.test'}).status_code==200
    assert client.get('/api/auth/me').status_code==401
    login(client);store.revoke(account.email)
    assert client.get('/api/auth/me').status_code==401


def test_explicit_existing_owner_import_preserves_id_and_refuses_conflicts(setup):
    store,_,_,client,provisioned,_,_=setup
    claims=VerifiedGoogleIdentity('alice@example.test','google-alice',True)
    store.import_existing_account(owner_id='u_existing_mail_owner',identity=claims)
    assert login(client).status_code==401  # Import alone is not an invitation.
    assert store.invite(claims.email).owner_id=='u_existing_mail_owner'
    assert login(client).status_code==303
    assert provisioned[0][0].owner_id=='u_existing_mail_owner'
    with pytest.raises(IdentityDenied):
        store.import_existing_account(owner_id='different-owner',identity=claims)


def test_provisioning_must_succeed_before_any_browser_session(setup):
    store,consent,broker,_,_,_,_=setup
    store.invite('alice@example.test')
    app=FastAPI()
    app.include_router(create_invited_auth_router(store,consent,broker,provision_account=lambda *args: False))
    with TestClient(app,base_url='https://gms.example.test') as client:
        response=login(client)
        assert response.status_code==503
        assert '__Host-gms_session' not in response.cookies
        assert client.get('/api/auth/me').status_code==401


def test_broker_http_redirects_and_unregistered_browser_destinations_are_refused(setup):
    from gmail_search.auth.invited_broker import BrokerUnavailable
    store,consent,_,client,_,_,_=setup
    store.invite('alice@example.test');login(client)
    session=client.cookies.get('__Host-gms_session')
    state=consent.begin(session)
    for response in [raw_response(302,headers={'location':'https://evil.test/steal'}),
                     raw_response(200,json={'url':'https://evil.test/v1/gmail/start?ticket='+'a'*64}),
                     raw_response(200,json={'url':'https://gmail-broker.example.test/v1/gmail/start?ticket='+'a'*64+'&owner_id=foreign'})]:
        calls=[]
        def transport(request):
            calls.append(str(request.url));return response
        with httpx.Client(trust_env=False,transport=httpx.MockTransport(transport),follow_redirects=True) as http:
            broker=BoundGmailBroker('https://gmail-broker.example.test',bearer='b'*48,signing_secret='h'*48,client=http)
            with pytest.raises(BrokerUnavailable):
                broker.start_consent(state)
        assert calls==['https://gmail-broker.example.test/v1/gmail/consents']


def test_reconnect_uses_new_credentials_and_keeps_the_same_browser_session(setup):
    store,_,_,client,_,posted,_=setup
    store.invite('alice@example.test');login(client)
    session=client.cookies.get('__Host-gms_session')
    connect(client);finish(client,posted)
    first=next(body for path,body in reversed(posted) if path.endswith('/consents'))
    client.post('/api/auth/disconnect-gmail',headers={'origin':'https://gms.example.test'})
    connect(client)
    second=next(body for path,body in reversed(posted) if path.endswith('/consents'))
    assert second['credential_generation']>first['credential_generation']
    assert second['invitation_generation']==first['invitation_generation']
    assert finish(client,posted).status_code==303
    assert client.cookies.get('__Host-gms_session')==session


def test_foreign_session_cannot_finish_another_users_consent(setup):
    store,_,_,client,_,posted,_=setup
    store.invite('alice@example.test');store.invite('bob@example.test')
    login(client);connect(client)
    login(client,email='bob@example.test',subject='google-bob')
    assert finish(client,posted).status_code==401
    assert client.get('/api/auth/gmail-status').json()=={'multi_tenant':True,'connect_method':'POST','connected':False}


def test_route_responses_and_auth_errors_never_cache_account_material(setup):
    _,_,_,client,_,_,_=setup
    for response in (client.get('/api/auth/me'),login(client)):
        assert response.headers['cache-control']=='private, no-store'
        assert response.headers['referrer-policy']=='no-referrer'


def test_cleanup_worker_retries_durable_failed_broker_revocation(setup):
    store,consent,broker,client,_,posted,fail=setup
    store.invite('alice@example.test');login(client);connect(client);finish(client,posted)
    fail[0]=True
    assert client.post('/api/auth/disconnect-gmail',headers={'origin':'https://gms.example.test'}).status_code==503
    fail[0]=False
    assert broker.drain_cleanup(consent)>=1
    assert consent.pending_cleanup()==()


def test_revocation_during_provisioning_cannot_admit_from_stale_generation(setup):
    store,consent,broker,_,_,_,_=setup
    store.invite('alice@example.test')
    def delayed_provision(account,claims):
        revocation=store.revoke(account.email)
        store.complete_revocation(revocation)
        store.invite(account.email)
        return True
    app=FastAPI()
    app.include_router(create_invited_auth_router(store,consent,broker,provision_account=delayed_provision))
    with TestClient(app,base_url='https://gms.example.test') as client:
        assert login(client).status_code==401
        assert client.get('/api/auth/me').status_code==401


def test_broker_request_ignores_injected_client_auth_cookies_params_and_headers(setup):
    store,consent,_,client,_,_,_=setup
    store.invite('alice@example.test');login(client)
    state=consent.begin(client.cookies.get('__Host-gms_session'))
    captured=[]
    def transport(request):
        captured.append(request)
        return raw_response(200,json={'url':'https://gmail-broker.example.test/v1/gmail/start?ticket='+'a'*64})
    with httpx.Client(trust_env=False,transport=httpx.MockTransport(transport),
        auth=('wrong','credentials'),cookies={'session':'private'},params={'owner_id':'foreign'},
        headers={'x-private':'must-not-leak'}) as http:
        broker=BoundGmailBroker('https://gmail-broker.example.test',bearer='b'*48,signing_secret='h'*48,client=http)
        broker.start_consent(state)
    request=captured[0]
    assert request.headers['authorization']=='Bearer '+'b'*48
    assert 'cookie' not in request.headers and 'x-private' not in request.headers
    assert str(request.url)=='https://gmail-broker.example.test/v1/gmail/consents'
    assert request.headers['accept-encoding']=='identity'


def raw_response(status_code, *, json=None, headers=None):
    import json as json_module
    return httpx.Response(status_code, headers=headers,
        stream=httpx.ByteStream(b'' if json is None else json_module.dumps(json).encode()))


@pytest.mark.parametrize('encoding', ['gzip', 'br', 'identity, gzip'])
def test_broker_rejects_encoded_response(encoding):
    with httpx.Client(trust_env=False, transport=httpx.MockTransport(
        lambda request: raw_response(200, json={}, headers={'content-encoding': encoding}))) as client:
        broker = BoundGmailBroker('https://gmail-broker.example.test', bearer='b'*32,
            signing_secret='s'*32, client=client)
        with pytest.raises(BrokerUnavailable):
            broker._post('/v1/gmail/consents', {})


@pytest.mark.parametrize('secret', ['é'*32, '\ud800'*32, 'a'*32+'\n'])
def test_broker_rejects_malformed_credentials(secret):
    with httpx.Client(trust_env=False) as client:
        with pytest.raises(BrokerUnavailable):
            BoundGmailBroker('https://gmail-broker.example.test', bearer=secret,
                signing_secret='s'*32, client=client)


def test_broker_rejects_environment_client():
    with httpx.Client() as client:
        with pytest.raises(BrokerUnavailable):
            BoundGmailBroker('https://gmail-broker.example.test', bearer='b'*32,
                signing_secret='s'*32, client=client)


def test_broker_raw_body_limit_closes_response():
    response = httpx.Response(200, stream=httpx.ByteStream(b' '*16385))
    with httpx.Client(trust_env=False, transport=httpx.MockTransport(lambda request: response)) as client:
        broker = BoundGmailBroker('https://gmail-broker.example.test', bearer='b'*32,
            signing_secret='s'*32, client=client)
        with pytest.raises(BrokerUnavailable):
            broker._post('/v1/gmail/consents', {})
        assert response.is_closed


def test_invited_identity_matches_existing_browser_auth_contract(setup):
    store,_,_,client,_,_,_ = setup
    owner = store.invite('alice@example.test')
    assert login(client).status_code == 303
    response = client.get('/api/auth/me')
    body = response.json()
    assert body.get('multi_tenant') is True
    assert body['user']['id'] == owner.owner_id


def test_invited_gmail_status_exposes_browser_onboarding_contract(setup):
    store,_,_,client,_,_,_ = setup
    store.invite('alice@example.test')
    login(client)
    assert client.get('/api/auth/gmail-status').json() == {
        'multi_tenant':True,'connected':False,'connect_method':'POST'}
