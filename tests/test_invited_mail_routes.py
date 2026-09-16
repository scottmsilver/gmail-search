from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from test_invited_run_routes import app_state, headers  # noqa: F401


def test_citation_routes_bind_session_and_recheck_after_read(app_state):  # noqa: F811
    from gmail_search.auth.mail_routes import create_mail_router
    s=app_state
    seen=[]
    async def thread(owner,thread_id,**page):
        seen.append((owner,thread_id,page))
        return {'thread_id':thread_id,'messages':[{'body_text':'PRIVATE'}]}
    async def lookup(owner,prefix):
        seen.append((owner,prefix))
        return {'thread_id':'abcde12345'}
    app=FastAPI();service=SimpleNamespace(thread=thread,lookup=lookup)
    app.include_router(create_mail_router(s.identities,service))
    with TestClient(app) as client:
        assert client.get('/api/thread/abcde12345').status_code==401
        assert client.get('/api/thread/abcde12345?owner_id=bob',headers=headers(s)).status_code==400
        assert client.get('/api/thread/abcde12345?body_offset=20000',headers=headers(s)).status_code==200
        assert seen[-1]==(s.accounts['alice'].owner_id,'abcde12345',{'body_offset':20000,'message_offset':0})
        assert client.get('/api/thread_lookup?cite_ref=abcde',headers=headers(s,'bob')).status_code==200
        assert seen[-1][0]==s.accounts['bob'].owner_id
        async def revoke(*args,**kwargs):
            s.identities.revoke_session(s.tokens['alice'])
            return {'messages':[{'body_text':'PRIVATE'}]}
        service.thread=revoke
        response=client.get('/api/thread/abcde12345',headers=headers(s))
        assert response.status_code==401 and 'PRIVATE' not in response.text


def test_attachment_metadata_and_download_are_session_owner_scoped(app_state):  # noqa: F811
    from gmail_search.auth.mail_routes import create_mail_router
    from gmail_search.gateway.browser_mail import BrowserAttachment
    s=app_state; seen=[]
    async def metadata(owner,attachment_id,**control):
        assert await control['check_active']() is True
        seen.append(('meta',owner,attachment_id))
        return dict(attachment_id=attachment_id,filename=owner+'.pdf',mime_type='application/pdf',
            size_bytes=12,message_id='message-one',thread_id='thread-one')
    async def download(owner,attachment_id,**control):
        assert await control['check_active']() is True
        seen.append(('download',owner,attachment_id))
        data=(owner+' private').encode()
        return BrowserAttachment(owner,attachment_id,owner+' report.pdf','application/pdf',
            len(data),'message-one','thread-one',data)
    service=SimpleNamespace(thread=None,lookup=None,attachment_metadata=metadata,attachment_download=download)
    app=FastAPI();app.include_router(create_mail_router(s.identities,service))
    with TestClient(app) as client:
        assert client.get('/api/attachment/7/meta').status_code==401
        response=client.get('/api/attachment/7/meta',headers=headers(s,'bob'))
        assert response.status_code==200
        assert response.json()==dict(attachment_id=7,filename=s.accounts['bob'].owner_id+'.pdf',
            mime_type='application/pdf',size_bytes=12,message_id='message-one',thread_id='thread-one')
        response=client.get('/api/attachment/7',headers=headers(s))
        assert response.status_code==200 and response.content==(s.accounts['alice'].owner_id+' private').encode()
        assert response.headers['content-type']=='application/octet-stream'
        assert response.headers['content-disposition'].startswith("attachment; filename*=UTF-8''")
        assert response.headers['content-security-policy']=="sandbox; default-src 'none'"
        assert response.headers['x-content-type-options']=='nosniff'
        assert response.headers['cache-control']=='private, no-store'
        assert seen[-1]==('download',s.accounts['alice'].owner_id,7)


@pytest.mark.parametrize('target',[
    '/api/attachment/0/meta','/api/attachment/01/meta','/api/attachment/9223372036854775808/meta',
    '/api/attachment/not-a-number/meta','/api/attachment/1/meta?owner_id=bob','/api/attachment/1?x=1',
])
def test_attachment_routes_reject_noncanonical_ids_and_parameters(app_state,target):  # noqa: F811
    from gmail_search.auth.mail_routes import create_mail_router
    async def forbidden(*args,**kwargs):pytest.fail('invalid request reached attachment service')
    service=SimpleNamespace(thread=None,lookup=None,attachment_metadata=forbidden,attachment_download=forbidden)
    app=FastAPI();app.include_router(create_mail_router(app_state.identities,service))
    with TestClient(app) as client:
        assert client.get(target,headers=headers(app_state)).status_code==400


def test_attachment_download_rejects_range_requests(app_state):  # noqa: F811
    from gmail_search.auth.mail_routes import create_mail_router
    async def forbidden(*args,**kwargs):pytest.fail('range request reached attachment service')
    service=SimpleNamespace(thread=None,lookup=None,attachment_metadata=forbidden,attachment_download=forbidden)
    app=FastAPI();app.include_router(create_mail_router(app_state.identities,service))
    request_headers={**headers(app_state),'Range':'bytes=0-1'}
    with TestClient(app) as client:
        assert client.get('/api/attachment/1',headers=request_headers).status_code==400


def test_attachment_rechecks_revocation_before_publishing_private_results(app_state):  # noqa: F811
    from gmail_search.auth.mail_routes import create_mail_router
    s=app_state
    async def metadata(owner,attachment_id,**control):
        s.identities.revoke_session(s.tokens['alice'])
        return {'attachment_id':attachment_id,'filename':'PRIVATE','mime_type':'application/pdf',
            'size_bytes':1,'message_id':'message','thread_id':'thread'}
    async def forbidden(*args,**kwargs):pytest.fail('unexpected download')
    service=SimpleNamespace(thread=None,lookup=None,attachment_metadata=metadata,attachment_download=forbidden)
    app=FastAPI();app.include_router(create_mail_router(s.identities,service))
    with TestClient(app) as client:
        response=client.get('/api/attachment/1/meta',headers=headers(s))
        assert response.status_code==401 and 'PRIVATE' not in response.text
