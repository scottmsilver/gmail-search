import assert from 'node:assert/strict';
import {test} from 'node:test';
import {tsImport} from 'tsx/esm/api';
process.env.PYTHON_API_URL = 'http://backend.test';
const mod = await tsImport(new URL('../app/api/agent/analyze/route.ts',import.meta.url).pathname,import.meta.url);
const route = mod.default ?? mod;
const originalFetch = globalThis.fetch;
function request(method,query,body) {
  const req = new Request('http://localhost/api/agent/analyze'+query,{method,
    headers:{cookie:'session=alice','Content-Type':'application/json'},
    ...(body ? {body:JSON.stringify(body)} : {})});
  req.nextUrl = new URL(req.url);
  return req;
}
test('replay forwards conversation binding and cursor without relaunch',async()=>{
  let call;
  globalThis.fetch = async (url,init)=>{call={url:String(url),init};return new Response('event: final\ndata: {}\n\n');};
  try {
    const response = await route.GET(request('GET','?session_id=run&conversation_id=conversation&after=12'));
    assert.equal(response.status,200);
    assert.equal(new URL(call.url).searchParams.get('conversation_id'),'conversation');
    assert.equal(new URL(call.url).searchParams.get('after'),'12');
  } finally {globalThis.fetch=originalFetch;}
});
test('Stop uses separate authenticated request with its own lifetime',async()=>{
  assert.equal(typeof route.DELETE,'function');
  let call;
  globalThis.fetch=async(url,init)=>{call={url:String(url),init};return Response.json({state:'cancelled'});};
  try {
    const result=await route.DELETE(request('DELETE','',{run_id:'run',conversation_id:'conversation'}));
    assert.equal(result.status,200);
    assert.equal(call.url,'http://backend.test/api/agent/analyze/run/cancel');
    assert.equal(call.init.method,'POST');
    assert.equal(call.init.headers.cookie,'session=alice');
    assert.deepEqual(JSON.parse(call.init.body),{conversation_id:'conversation'});
    assert.equal((await result.json()).state,'cancelled');
  } finally {globalThis.fetch=originalFetch;}
});
