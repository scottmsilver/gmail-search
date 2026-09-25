// Run from web/: node --import tsx scripts/test-deep-route.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, writeFile, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { tsImport } from 'tsx/esm/api';
process.env.PYTHON_API_URL = 'http://backend.test';
process.env.GMAIL_CHAT_LOG_DIR = await mkdtemp(join(tmpdir(),'deep-route-'));
const mod = await tsImport(new URL('../app/api/chat/route.ts',import.meta.url).pathname,import.meta.url);
const {POST} = mod.default ?? mod;
const messages = [{id:'u1',role:'user',parts:[{type:'text',text:'Investigate'}]}];
const request = body => new Request('http://localhost/api/chat',{method:'POST',headers:{'Content-Type':'application/json',cookie:'session=owner'},body:JSON.stringify(body)});
const identity = (id='owner') => Response.json({multi_tenant:true,user:{id,email:`${id}@example.test`}});
const originalFetch = globalThis.fetch;
test('stopping browser chat aborts the Python analysis stream', async () => {
  const browser = new AbortController();
  const req = new Request('http://localhost/api/chat', {method:'POST',
    headers:{'Content-Type':'application/json',cookie:'session=owner'},
    body:JSON.stringify({messages}), signal:browser.signal});
  let upstreamSignal, bodyController, abortObserved = false;
  let entered;
  const started = new Promise(resolve => { entered = resolve; });
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/api/auth/me')) return identity();
    upstreamSignal = init.signal;
    const body = new ReadableStream({start(controller) {
      bodyController = controller;
      init.signal?.addEventListener('abort', () => {
        abortObserved = true;
        controller.error(new DOMException('Request cancelled','AbortError'));
      }, {once:true});
    }});
    entered();
    return new Response(body);
  };
  try {
    const response = await POST(req);
    await started;
    assert.equal(upstreamSignal, req.signal);
    browser.abort();
    await response.text();
    assert.equal(abortObserved, true);
  } finally {
    try { bodyController?.error(new Error('test cleanup')); } catch {}
    globalThis.fetch = originalFetch;
  }
});
test('deep request writes the advertised log with session and terminal error', async () => {
  globalThis.fetch = async url => String(url).endsWith('/api/auth/me') ? identity() : new Response([
    'event: session', 'data: {"session_id":"real-session"}', '',
    'event: error', 'data: {"payload":{"message":"MALFORMED_FUNCTION_CALL"}}', '', '',
  ].join('\n'));
  try {
    const response = await POST(request({messages}));
    const output = await response.text();
    const frames = output.split('\n').filter(line => line.startsWith('data: {')).map(line => JSON.parse(line.slice(6)));
    const debug = frames.find(frame => frame.type === 'data-debug-id').data;
    const entries = (await readFile(debug.log_path, 'utf8')).trim().split('\n').map(JSON.parse);
    assert.equal(entries[0].kind, 'request');
    assert.equal(entries[0].owner_id, 'owner');
    assert.ok(entries.some(e => e.data.session_id === 'real-session'));
    assert.ok(entries.some(e => e.kind === 'error' && JSON.stringify(e.data).includes('MALFORMED_FUNCTION_CALL')));
    assert.equal(entries.at(-1).kind, 'done');
  } finally { globalThis.fetch = originalFetch; }
});
const sse = (id) => `event: session\ndata: {"session_id":"${id}"}\n\nevent: final\ndata: {"payload":{"text":"Answer ${id}"}}\n\n`;
test('legacy deep:false requests still run deep analysis and preserve backend/model', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {if(String(url).endsWith('/api/auth/me')) return identity(); calls.push({url,init}); return new Response(sse('single'));};
  try {
    const response = await POST(request({messages,deep:false,deep_backend:'pi',model:'openrouter/meta/muse-spark-1.3'}));
    assert.match(await response.text(), /Answer single/);
    assert.equal(calls.length,1);
    assert.match(calls[0].url,/\/api\/agent\/analyze$/);
    const body = JSON.parse(calls[0].init.body);
    assert.equal(body.backend,'pi');
    assert.equal(body.model,'openrouter/meta/muse-spark-1.3');
  } finally {globalThis.fetch = originalFetch;}
});
test('battle invokes two isolated deep runs and saves both answers once after initial save', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    if(String(url).endsWith('/api/auth/me')) return identity();
    const body = JSON.parse(init.body); calls.push({url,init,body});
    if (init.method === 'PUT') return Response.json({ok:true});
    return new Response(sse(`session-${calls.length}`));
  };
  try {
    const response = await POST(request({messages,battle:true,conversation_id:'conv1'}));
    const text = await response.text();
    assert.match(text,/data-battle/);
    const runs = calls.filter(c=>c.init.method==='POST');
    assert.equal(runs.length,2);
    assert.ok(runs.every(c=>c.body.conversation_id === undefined && c.body.backend && c.init.headers.cookie==='session=owner'));
    assert.equal(runs[0].body.question,runs[1].body.question);
    const saves = calls.filter(c=>c.init.method==='PUT');
    assert.equal(saves.length,2);
    const data = saves[1].body.messages.at(-1).parts[0].data;
    assert.ok(data.answer_a && data.answer_b && data.session_a !== data.session_b);
    assert.equal(data.running_a,false);
    assert.equal(data.running_b,false);
  } finally {globalThis.fetch = originalFetch;}
});
test('failed initial save does not launch chargeable analyses', async () => {
  let calls = 0;
  globalThis.fetch = async url => {if(String(url).endsWith('/api/auth/me')) return identity(); calls++; return new Response('',{status:403});};
  try {
    const response = await POST(request({messages,battle:true,conversation_id:'not-owned'}));
    assert.equal(response.status,502);
    assert.equal(calls,1);
  } finally {globalThis.fetch = originalFetch;}
});
const logMod = await tsImport(new URL('../lib/chatLog.ts',import.meta.url).pathname,import.meta.url);
const { ChatLogger } = logMod.default ?? logMod;
test('logging failure cannot prevent final battle persistence', async () => {
  let saves = 0;
  const log = ChatLogger.prototype.log;
  ChatLogger.prototype.log = async () => {throw new Error('Disk unavailable');};
  globalThis.fetch = async (url, init) => {
    if(String(url).endsWith('/api/auth/me')) return identity();
    if (init.method === 'PUT') {saves++; return Response.json({ok:true});}
    return new Response(sse('side'));
  };
  try {
    const response = await POST(request({messages,battle:true,conversation_id:'conv2'}));
    await response.text();
    assert.equal(saves,2);
  } finally {globalThis.fetch = originalFetch; ChatLogger.prototype.log = log;}
});
test('saved removed backend migrates to Pi with a valid model', async () => {
  globalThis.window = {localStorage:{getItem:()=>JSON.stringify({deepBackend:'claude_native',model:'opus'}),setItem:()=>{}}};
  try {
    const module = await tsImport(new URL('../lib/chatSettings.ts',import.meta.url).pathname,import.meta.url);
    const settings = (module.default ?? module).getChatSettings();
    assert.equal(settings.deepBackend,'pi');
    assert.equal(settings.model,'google/gemini-3.8-flash');
  } finally {delete globalThis.window;}
});

const logRouteMod = await tsImport(new URL('../app/api/log/[id]/route.ts',import.meta.url).pathname,import.meta.url);
const logGET = (logRouteMod.default ?? logRouteMod).GET;
test('anonymous and legacy anonymous chat fail before reading a body or writing a log', async () => {
  try {
    for (const auth of [()=>new Response('',{status:401}),()=>Response.json({multi_tenant:false,user:null})]) {
      for (const battle of [false,true]) {
        const before = await readdir(process.env.GMAIL_CHAT_LOG_DIR);
        let calls=0;
        globalThis.fetch=async url=>{calls++; assert.match(String(url),/\/api\/auth\/me$/); return auth();};
        const req=request({messages,battle});
        req.json=async()=>{throw new Error('unauthenticated body was read');};
        const response=await POST(req);
        assert.equal(response.status,401);
        assert.equal(calls,1);
        assert.deepEqual(await readdir(process.env.GMAIL_CHAT_LOG_DIR),before);
      }
    }
  } finally {globalThis.fetch=originalFetch;}
});
test('log download authenticates and rejects foreign or legacy owner records without disclosing bytes', async () => {
  const id='abcdef1234567890';
  const secret='PRIVATE MAIL MARKER';
  const file=join(process.env.GMAIL_CHAT_LOG_DIR,`${id}.jsonl`);
  try {
    for (const [owner,viewer,status] of [['owner',null,401],['owner','other',404],[null,'owner',404],['owner','owner',200]]) {
      await writeFile(file,JSON.stringify({owner_id:owner,kind:'request',data:{secret}})+'\n');
      globalThis.fetch=async()=>viewer ? identity(viewer) : new Response('',{status:401});
      const response=await logGET(new Request('http://localhost/api/log/'+id),{params:Promise.resolve({id})});
      assert.equal(response.status,status);
      assert.equal(response.headers.get('cache-control'),'private, no-store');
      const text=await response.text();
      if(status===200) assert.match(text,/PRIVATE MAIL MARKER/); else assert.ok(!text.includes(secret));
    }
  } finally {globalThis.fetch=originalFetch;}
});
for (const route of ['artifact','attachment']) {
  test(`${route} active content cannot execute on the application origin`,async()=>{
    const m=await tsImport(new URL(`../app/api/${route}/[id]/route.ts`,import.meta.url).pathname,import.meta.url);
    try {
      for(const mime of ['text/html','image/svg+xml','application/xhtml+xml']) {
        globalThis.fetch=async()=>new Response('<script>attack()</script>',{headers:{'content-type':mime,'content-disposition':'inline; filename="evil.html"','referrer-policy':'no-referrer'}});
        const response=await (m.default??m).GET(new Request('http://localhost/api/'+route+'/123'),{params:Promise.resolve({id:'123'})});
        assert.match(response.headers.get('content-disposition'),/^attachment/);
        assert.equal(response.headers.get('x-content-type-options'),'nosniff');
        assert.equal(response.headers.get('cache-control'),'private, no-store');
        assert.match(response.headers.get('content-security-policy'),/sandbox/);
        assert.equal(response.headers.get('referrer-policy'),'no-referrer');
      }
    } finally {globalThis.fetch=originalFetch;}
  });
}

test('chat rejects a foreign Origin before authentication or body processing',async()=>{
  const req=request({messages}); req.headers.set('origin','https://evil.example');
  try {
    globalThis.fetch=async()=>{throw new Error('must not reach backend');};
    assert.equal((await POST(req)).status,403);
  } finally {globalThis.fetch=originalFetch;}
});
test('public chat forwards the validated Origin to each internal mutation',async()=>{
  const prior=process.env.GMS_PUBLIC_ORIGIN;
  process.env.GMS_PUBLIC_ORIGIN='https://mail.example.test';
  try {
    for(const battle of [false]) {
      let mutations=0;
      globalThis.fetch=async(url,init)=>{
        if(String(url).endsWith('/api/auth/me')) return identity();
        mutations++;
        assert.equal(init.headers.origin,'https://mail.example.test');
        return init.method==='PUT' ? Response.json({ok:true}) : new Response(sse('public'));
      };
      const req=request({messages,battle,conversation_id:'publicconv'});
      req.headers.set('origin','https://mail.example.test'); req.headers.set('host','mail.example.test');
      const response=await POST(req);
      const text=await response.text();
      assert.ok(!text.includes('AssertionError'));
      assert.match(text,battle ? /Answer public/ : /Answer public/);
      assert.ok(mutations >= 2);
    }
  } finally {
    if(prior===undefined) delete process.env.GMS_PUBLIC_ORIGIN; else process.env.GMS_PUBLIC_ORIGIN=prior;
    globalThis.fetch=originalFetch;
  }
});
test('API middleware denies unsafe foreign or missing public origins and marks private responses',async()=>{
  const m=await tsImport(new URL('../middleware.ts',import.meta.url).pathname,import.meta.url);
  const middleware=(m.default??m).middleware;
  const prior=process.env.GMS_PUBLIC_ORIGIN;
  try {
    process.env.GMS_PUBLIC_ORIGIN='https://mail.example.test';
    for(const method of ['POST','PUT','DELETE','PATCH']) {
      for(const origin of [null,'https://sibling.example.test','null']) {
        const req=new Request('http://localhost/api/conversations/abcdef',{method,headers:origin?{origin}:{}});
        assert.equal(middleware(req).status,403);
      }
      const req=new Request('http://localhost/api/conversations/abcdef',{method,headers:{origin:'https://mail.example.test'}});
      assert.equal(middleware(req).headers.get('cache-control'),'private, no-store');
    }
  } finally {if(prior===undefined)delete process.env.GMS_PUBLIC_ORIGIN;else process.env.GMS_PUBLIC_ORIGIN=prior;}
});
// Battles were unavailable on the public origin, so the vote proxy was denied
// outright at the boundary. Capable owners may battle now, so the boundary
// admits the route and authentication becomes the gate instead -- an
// unauthenticated vote must still never reach the backend or read a body.
test('public vote proxy still authenticates before backend or body reads',async()=>{
  const m=await tsImport(new URL('../app/api/battle/vote/route.ts',import.meta.url).pathname,import.meta.url);
  const vote=(m.default??m).POST;
  process.env.GMS_PUBLIC_ORIGIN='https://mail.example.test';
  try {
    globalThis.fetch=async()=>{throw new Error('must not reach backend');};
    const req=new Request('https://mail.example.test/api/battle/vote',{method:'POST',headers:{host:'mail.example.test',origin:'https://mail.example.test'},body:'{}'});
    assert.equal((await vote(req)).status,401);
  } finally {delete process.env.GMS_PUBLIC_ORIGIN;globalThis.fetch=originalFetch;}
});
test('private HTTPS proxy origin uses forwarded scheme while public origin ignores it',async()=>{
  const m=await tsImport(new URL('../lib/originSecurity.ts',import.meta.url).pathname,import.meta.url);
  const guard=(m.default??m).rejectUnsafeOrigin;
  const prior=process.env.GMS_PUBLIC_ORIGIN;
  try {
    delete process.env.GMS_PUBLIC_ORIGIN;
    const req=new Request('http://localhost/api/chat',{method:'POST',headers:{host:'mail.internal.test',origin:'https://mail.internal.test','x-forwarded-proto':'https'}});
    assert.equal(guard(req),null);
    process.env.GMS_PUBLIC_ORIGIN='https://mail.public.test';
    assert.equal(guard(req).status,403);
  } finally {if(prior===undefined)delete process.env.GMS_PUBLIC_ORIGIN;else process.env.GMS_PUBLIC_ORIGIN=prior;}
});
test('empty public origin fails closed instead of enabling private mode',async()=>{
  const m=await tsImport(new URL('../lib/originSecurity.ts',import.meta.url).pathname,import.meta.url);
  const guard=(m.default??m).rejectUnsafeOrigin;
  const prior=process.env.GMS_PUBLIC_ORIGIN;
  try {
    process.env.GMS_PUBLIC_ORIGIN='';
    assert.equal(guard(new Request('http://localhost/api/chat',{method:'POST'})).status,403);
    assert.equal(guard(new Request('http://localhost/api/chat',{method:'POST',headers:{origin:''}})).status,403);
  } finally {if(prior===undefined)delete process.env.GMS_PUBLIC_ORIGIN;else process.env.GMS_PUBLIC_ORIGIN=prior;}
});
test('auth callback middleware suppresses outgoing referrers',async()=>{
  const m=await tsImport(new URL('../middleware.ts',import.meta.url).pathname,import.meta.url);
  const response=(m.default??m).middleware(new Request('https://mail.example.test/api/auth/callback?token=secret'));
  assert.equal(response.headers.get('referrer-policy'),'no-referrer');
});
test('email document policy blocks remote resources while preserving inline styles',async()=>{
  const source=await readFile(new URL('../components/EmailBody.tsx',import.meta.url),'utf8');
  assert.match(source,/http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"/);
  assert.match(source,/referrerPolicy="no-referrer"/);
});

test('public chat ignores runtime/model override, suppresses debug paths, and refuses battle',async()=>{
  process.env.GMS_PUBLIC_ORIGIN='https://mail.example.test';
  let runs=0;
  const publicRequest=body=>{const req=request(body);req.headers.set('host','mail.example.test');req.headers.set('origin','https://mail.example.test');return req;};
  try {
    globalThis.fetch=async(url,init)=>{
      if(String(url).endsWith('/api/auth/me')) return identity();
      runs++;
      const body=JSON.parse(init.body);
      assert.equal(body.backend,'pi');
      assert.equal(body.model,undefined);
      return new Response(sse('restricted'));
    };
    const response=await POST(publicRequest({messages,deep_backend:'claude_code',model:'opus'}));
    const text=await response.text();
    assert.match(text,/Answer restricted/); assert.ok(!text.includes('data-debug-id')); assert.equal(runs,1);
    assert.equal((await POST(publicRequest({messages,battle:true}))).status,400); assert.equal(runs,1);
    assert.equal((await POST(publicRequest({messages:[null]}))).status,400); assert.equal(runs,1);
  } finally {delete process.env.GMS_PUBLIC_ORIGIN;globalThis.fetch=originalFetch;}
});
test('worker session metadata reaches browser Stop controls', async () => {
  globalThis.fetch = async url => String(url).endsWith('/api/auth/me') ? identity() : new Response([
    'event: session', 'data: {"session_id":"abc123","conversation_id":"conversation","supports_cancel":true}', '',
    'event: final', 'data: {"payload":{"text":"Saved answer"}}', '',
    'event: persist_ok', 'data: {}', '', '',
  ].join('\n'));
  try {
    const response = await POST(request({messages}));
    const output = await response.text();
    const frames = output.split('\n').filter(line => line.startsWith('data: {')).map(line => JSON.parse(line.slice(6)));
    assert.deepEqual(frames.find(frame => frame.type === 'data-agent-run')?.data,
      {runId:'abc123',conversationId:'conversation'});
    assert.deepEqual(frames.find(frame => frame.type === 'data-agent-run-finished')?.data, {runId:'abc123'});
  } finally {globalThis.fetch = originalFetch;}
});
test('worker-owned transcript is never replaced when its stream loses completion frames', async () => {
  for (const body of ['', 'event: final\ndata: {"payload":{"text":"Already saved"}}\n\n']) {
    const writes=[];
    globalThis.fetch=async(url,init)=>{
      if(String(url).endsWith('/api/auth/me')) return identity();
      if(init?.method==='PUT') {writes.push(JSON.parse(init.body));return Response.json({ok:true});}
      return new Response(body,{headers:{'X-GMS-Transcript-Owner':'server'}});
    };
    try {
      const response=await POST(request({messages,conversation_id:'conversation'}));
      await response.text();
      assert.equal(writes.length,1,'only the initial user-turn save may occur');
      assert.deepEqual(writes[0].messages,messages.map(({role,parts})=>({role,parts})));
    } finally {globalThis.fetch=originalFetch;}
  }
});

// ── capability, not deployment ───────────────────────────────────────────────
// The route used to refuse battles whenever GMS_PUBLIC_ORIGIN was set, so the
// public build was locked down for everyone on it. It now follows the same
// server-reported capability the UI renders from: /api/auth/me's
// capabilities.full_runtime, which is derived from the authenticated session
// and never from the request body.

// With GMS_PUBLIC_ORIGIN set, publicRoute enforces same-origin, so these
// requests must carry the matching Origin or they are refused before the
// capability check is ever reached.
const PUBLIC_ORIGIN = 'https://public.example.test';
const publicRequest = body => new Request('http://localhost/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json', cookie: 'session=owner',
            origin: PUBLIC_ORIGIN, host: 'public.example.test'},
  body: JSON.stringify(body),
});

const capableIdentity = (id='owner') => Response.json({
  multi_tenant: true, user: {id, email: `${id}@example.test`},
  capabilities: {full_runtime: true},
});

test('public deployment still refuses battles for an ordinary caller', async () => {
  process.env.GMS_PUBLIC_ORIGIN = PUBLIC_ORIGIN;
  globalThis.fetch = async url => String(url).endsWith('/api/auth/me') ? identity() : new Response(sse('single'));
  try {
    const response = await POST(publicRequest({messages, battle: true, conversation_id: 'conv-restricted'}));
    assert.equal(response.status, 400);
    assert.match(await response.text(), /Battle mode is unavailable/);
  } finally {
    delete process.env.GMS_PUBLIC_ORIGIN;
    globalThis.fetch = originalFetch;
  }
});

test('public deployment allows battles for a caller the server reports as capable', async () => {
  process.env.GMS_PUBLIC_ORIGIN = PUBLIC_ORIGIN;
  globalThis.fetch = async url => String(url).endsWith('/api/auth/me') ? capableIdentity() : new Response(sse('single'));
  try {
    const response = await POST(publicRequest({messages, battle: true, conversation_id: 'conv-capable'}));
    assert.notEqual(response.status, 400);
  } finally {
    delete process.env.GMS_PUBLIC_ORIGIN;
    globalThis.fetch = originalFetch;
  }
});

test('an admitted worker run shows a starting step before the VM reports in', async () => {
  // The guest's first event waits on the VM booting; without this the bubble
  // sits empty for seconds after the user hits send.
  globalThis.fetch = async (url) => String(url).endsWith('/api/auth/me') ? identity()
    : new Response('event: session\ndata: {"session_id":"run-1","conversation_id":"c1","supports_cancel":true}\n\n'
        + 'event: final\ndata: {"payload":{"text":"Answer"}}\n\n');
  try {
    const text = await (await POST(request({messages, conversation_id:'c1'}))).text();
    const stages = text.split('\n').filter(l => l.includes('"data-deep-stage"'));
    assert.ok(stages.length > 0, 'a deep stage is emitted');
    assert.ok(stages[0].includes('"state":"starting"'), `first stage is starting, got ${stages[0].slice(0,200)}`);
  } finally { globalThis.fetch = originalFetch; }
});

function workerStream(frames) {
  return async (url) => String(url).endsWith('/api/auth/me') ? identity()
    : new Response('event: session\ndata: {"session_id":"run-1","conversation_id":"c1","supports_cancel":true}\n\n'
        + frames.map(([kind, payload]) => `event: ${kind}\ndata: ${JSON.stringify({payload})}\n\n`).join(''));
}

function textParts(body) {
  const lines = body.split('\n').filter(l => l.startsWith('data: {'));
  const events = lines.map(l => JSON.parse(l.slice(6)));
  return {
    starts: events.filter(e => e.type === 'text-start').map(e => e.id),
    text: events.filter(e => e.type === 'text-delta').map(e => e.delta).join(''),
  };
}

test('answer text streams as it is written and the matching final is not repeated', async () => {
  globalThis.fetch = workerStream([['answer_delta', {text: 'Three '}], ['answer_delta', {text: '4000W heaters.'}],
    ['final', {text: 'Three 4000W heaters.'}]]);
  try {
    const parts = textParts(await (await POST(request({messages, conversation_id:'c1'}))).text());
    assert.deepEqual(parts.starts, ['deep-stream-' + parts.starts[0].split('deep-stream-')[1]]);
    assert.equal(parts.text, 'Three 4000W heaters.');
  } finally { globalThis.fetch = originalFetch; }
});

test('a final that differs from the streamed text is still shown', async () => {
  globalThis.fetch = workerStream([['answer_delta', {text: 'Draft'}], ['final', {text: 'Final answer'}]]);
  try {
    const parts = textParts(await (await POST(request({messages, conversation_id:'c1'}))).text());
    assert.equal(parts.starts.length, 2);
    assert.ok(parts.text.endsWith('Final answer'));
  } finally { globalThis.fetch = originalFetch; }
});
