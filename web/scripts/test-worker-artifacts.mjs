import assert from 'node:assert/strict';
import {test} from 'node:test';
import {tsImport} from 'tsx/esm/api';
const load=async path=>{const m=await tsImport(new URL(path,import.meta.url).pathname,import.meta.url);return m.default??m;};
const {linkifyRefs}=await load('../lib/linkifyRefs.ts');
const {GET}=await load('../app/api/artifact/[id]/route.ts');
const id='a'.repeat(32);
const original=globalThis.fetch;
process.env.PYTHON_API_URL='http://backend.test';
test('opaque artifact citation preserves the complete ID',()=>{
  assert.equal(linkifyRefs(`[art:${id}]`,[]),`[${id}](art://${id})`);
  assert.equal(linkifyRefs('[art:12]',[]),'[12](art://12)');
});
test('worker artifact proxy requires conversation and forwards scoped download',async()=>{
  process.env.GMS_FULL_WORKER_ROUTES='1';
  let call;
  globalThis.fetch=async(url,init)=>{call={url:String(url),init};return new Response('report',{headers:{'Content-Type':'text/csv'}});};
  const ctx={params:Promise.resolve({id})};
  try {
    const missing=await GET(new Request(`http://localhost/api/artifact/${id}`),ctx);
    assert.equal(missing.status,400);
    assert.equal(call,undefined);
    const result=await GET(new Request(`http://localhost/api/artifact/${id}?conversation_id=conversation`,{headers:{cookie:'session=alice'}}),ctx);
    assert.equal(result.status,200);
    assert.equal(call.url,`http://backend.test/api/agent-artifacts/${id}?conversation_id=conversation`);
    assert.equal(call.init.headers.cookie,'session=alice');
    assert.equal(result.headers.get('cache-control'),'private, no-store');
    assert.match(result.headers.get('content-security-policy'),/sandbox/);
  } finally {globalThis.fetch=original;delete process.env.GMS_FULL_WORKER_ROUTES;}
});
