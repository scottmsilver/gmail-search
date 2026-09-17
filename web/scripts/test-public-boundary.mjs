import assert from 'node:assert/strict';
import {test} from 'node:test';
import {tsImport} from 'tsx/esm/api';
const mod = await tsImport(new URL('../lib/publicBoundary.ts', import.meta.url).pathname, import.meta.url);
const {publicRoute, publicRequestDenied} = mod.default ?? mod;
const origin = 'https://gms.oursilverfamily.com';
process.env.PYTHON_API_URL = 'http://backend.test';
const request = (path, method='GET', extra={}, body) => new Request(origin+path, {method, headers:{host:'gms.oursilverfamily.com',origin,...extra}, ...(body ? {body,duplex:'half'} : {})});
test('OAuth return navigations reach UI pages while cross-site API and fetch requests stay blocked', ()=>{
 process.env.GMS_PUBLIC_ORIGIN=origin;
 const navigation = (path, method='GET', extra={}) => new Request(origin+path, {method, headers:{host:'gms.oursilverfamily.com','sec-fetch-site':'cross-site','sec-fetch-mode':'navigate','sec-fetch-dest':'document',...extra}});
 try {
  for (const site of ['same-site','cross-site']) for (const method of ['GET','HEAD']) for (const path of ['/','/search','/settings']) {
   assert.equal(publicRequestDenied(navigation(path,method,{'sec-fetch-site':site})),null);
  }
  for (const path of ['/api/search','/api/conversations','/_next/static/app.js','/favicon.ico','/admin']) assert.equal(publicRequestDenied(navigation(path)).status,403);
  for (const method of ['POST','PUT','DELETE']) assert.equal(publicRequestDenied(navigation('/',method,{origin})).status,403);
  for (const extra of [
   {'sec-fetch-mode':'cors'}, {'sec-fetch-mode':''}, {'sec-fetch-mode':'navigate, navigate'},
   {'sec-fetch-dest':'iframe'}, {'sec-fetch-dest':''}, {'sec-fetch-dest':'document, document'},
   {'sec-fetch-site':'cross-site, same-origin'}, {'sec-fetch-site':'same-origin, cross-site'},
   {origin:'https://evil.test'}, {host:'evil.test'}, {'x-user-id':'spoofed'},
  ]) assert.equal(publicRequestDenied(navigation('/','GET',extra)).status,403);
 } finally { delete process.env.GMS_PUBLIC_ORIGIN; }
});
test('public boundary defaults deny, exact origin/host and internal headers; private unchanged', async()=>{
 process.env.GMS_PUBLIC_ORIGIN=origin;
 // '/api/battle/stats' was in this deny list while battles were unavailable on
 // the public origin. Capable owners may battle now, so the two user-scoped
 // battle endpoints are allowlisted and asserted separately below; everything
 // else here still defaults to denied.
 for(const path of ['/api/admin/users','/api/jobs/frontfill','/api/log/abc','/api/auth/whoami','/api/agent/analyze','/api/unknown']) assert.equal(publicRequestDenied(request(path)).status,404);
 assert.equal(publicRequestDenied(request('/api/chat','GET')).status,404);
 assert.equal(publicRequestDenied(request('/api/search','GET',{host:'evil.test'})).status,403);
 assert.equal(publicRequestDenied(request('/api/search','GET',{origin:'https://evil.test'})).status,403);
 for (const header of ['x-middleware-subrequest','x-user-id','authorization']) assert.equal(publicRequestDenied(request('/api/search','GET',{[header]:'spoofed'})).status,403);
 assert.equal(publicRequestDenied(request('/api/auth/callback','GET',{'sec-fetch-site':'cross-site'})),null);
 process.env.GMS_PUBLIC_ORIGIN='';
 assert.equal(publicRequestDenied(request('/api/auth/login')).status,503);
 delete process.env.GMS_PUBLIC_ORIGIN;
 assert.equal(publicRequestDenied(request('/api/admin/users')),null);
});
test('protected handlers authenticate before body, reject chunked oversize, and secure responses', async()=>{
 process.env.GMS_PUBLIC_ORIGIN=origin;
 const original=globalThis.fetch; let called=0;
 const route=publicRoute(async()=>{called++;return Response.json({ok:true});});
 try {
  globalThis.fetch=async()=>Response.json({multi_tenant:false,user:null});
  assert.equal((await route(request('/api/chat','POST',{},'{}'))).status,401); assert.equal(called,0);
  globalThis.fetch=async()=>Response.json({multi_tenant:true,user:{id:'owner'}});
  const stream=new ReadableStream({start(c){c.enqueue(new Uint8Array(262145));c.close();}});
  assert.equal((await route(request('/api/chat','POST',{},stream))).status,413); assert.equal(called,0);
  const response=await route(request('/api/chat','POST',{},'{}'));
  assert.equal(called,1); assert.equal(response.headers.get('cache-control'),'private, no-store'); assert.equal(response.headers.get('referrer-policy'),'no-referrer');
 } finally { globalThis.fetch=original; delete process.env.GMS_PUBLIC_ORIGIN; }
});
test('slow bodies time out and cancel source; download sandbox policy survives', async()=>{
 process.env.GMS_PUBLIC_ORIGIN=origin;
 const originalFetch=globalThis.fetch, originalTimer=globalThis.setTimeout;
 let cancelled=false;
 try {
  globalThis.fetch=async()=>Response.json({multi_tenant:true,user:{id:'owner'}});
  globalThis.setTimeout=(fn,ms,...args)=>originalTimer(fn,ms===15000?1:ms,...args);
  const stream=new ReadableStream({cancel(){cancelled=true;}});
  const route=publicRoute(async()=>new Response('ok',{headers:{'Content-Security-Policy':"sandbox; default-src 'none'"}}));
  assert.equal((await route(request('/api/chat','POST',{},stream))).status,408);
  await new Promise(resolve=>originalTimer(resolve,5));
  assert.equal(cancelled,true);
  const response=await route(request('/api/attachment/id'));
  assert.match(response.headers.get('content-security-policy'),/sandbox/);
  assert.match(response.headers.get('content-security-policy'),/frame-ancestors 'none'/);
  assert.equal(response.headers.get('x-frame-options'),'DENY');
 } finally {globalThis.fetch=originalFetch;globalThis.setTimeout=originalTimer;delete process.env.GMS_PUBLIC_ORIGIN;}
});
test('full-worker control routes require explicit rollout flag and retain authentication',async()=>{
 process.env.GMS_PUBLIC_ORIGIN=origin;
 process.env.GMS_FULL_WORKER_ROUTES='1';
 const original=globalThis.fetch;
 let called=false;
 const route=publicRoute(async()=>{called=true;return Response.json({ok:true});});
 try {
  for (const method of ['GET','DELETE']) assert.equal(publicRequestDenied(request('/api/agent/analyze',method)),null);
  assert.equal(publicRequestDenied(request('/api/agent/analyze','POST')).status,404);
  assert.equal(publicRequestDenied(request('/api/agent/analyze','DELETE',{origin:'https://evil.test'})).status,403);
  globalThis.fetch=async()=>Response.json({user:null});
  assert.equal((await route(request('/api/agent/analyze','DELETE',{},'{}'))).status,401);
  assert.equal(called,false);
 } finally {globalThis.fetch=original;delete process.env.GMS_FULL_WORKER_ROUTES;delete process.env.GMS_PUBLIC_ORIGIN;}
});

// Battles became available to capable owners on the public origin, but their
// vote and stats endpoints were never added to the boundary allowlist, so a
// battle could run and then 404 the moment anyone voted on it. Reaching the
// endpoint is not the same as being allowed to battle: the chat route still
// gates that on capability, and both endpoints still require a session.
//
// These set GMS_PUBLIC_ORIGIN themselves. The tests above delete it in their
// finally blocks, and without it publicRequestDenied returns null immediately,
// so an allowlist test written without it passes whether or not the route is
// actually permitted.
test('battle vote and stats are reachable on the public origin', () => {
  process.env.GMS_PUBLIC_ORIGIN = origin;
  try {
    assert.equal(publicRequestDenied(request('/api/battle/vote', 'POST')), null);
    assert.equal(publicRequestDenied(request('/api/battle/stats')), null);
  } finally { delete process.env.GMS_PUBLIC_ORIGIN; }
});

test('battle endpoints still refuse the wrong method', () => {
  process.env.GMS_PUBLIC_ORIGIN = origin;
  try {
    assert.equal(publicRequestDenied(request('/api/battle/vote'))?.status, 404);
    assert.equal(publicRequestDenied(request('/api/battle/stats', 'POST'))?.status, 404);
  } finally { delete process.env.GMS_PUBLIC_ORIGIN; }
});
