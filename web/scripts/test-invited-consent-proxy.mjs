import assert from 'node:assert/strict';
import {test} from 'node:test';
import {tsImport} from 'tsx/esm/api';
const load=async path=>{const m=await tsImport(new URL(path,import.meta.url).pathname,import.meta.url);return m.default??m;};
const connect=await load('../app/api/auth/connect-gmail/route.ts');
const origin='https://gms.example.test';
process.env.PYTHON_API_URL='http://backend.test';
const original=globalThis.fetch;
test('invited Connect uses authenticated POST and preserves consent cookie and redirect',async()=>{
  assert.equal(typeof connect.POST,'function');
  process.env.GMS_PUBLIC_ORIGIN=origin;
  process.env.GMS_FULL_WORKER_ROUTES='1';
  let call;
  globalThis.fetch=async(url,init)=>{
    if(String(url).endsWith('/api/auth/me')) return Response.json({multi_tenant:true,user:{id:'alice'}});
    call={url:String(url),init};
    return new Response(null,{status:303,headers:{location:'https://broker.example.test/consent','Set-Cookie':'__Host-gms_gmail_state=synthetic; Secure; HttpOnly; SameSite=Lax; Path=/'}});
  };
  try {
    const req=new Request(origin+'/api/auth/connect-gmail',{method:'POST',headers:{host:'gms.example.test',origin,cookie:'session=alice'}});
    req.nextUrl=new URL(req.url);
    const response=await connect.POST(req);
    assert.equal(response.status,303);
    assert.equal(call.init.method,'POST');
    assert.equal(call.init.redirect,'manual');
    assert.equal(call.init.headers.origin,origin);
    assert.equal(call.init.headers.cookie,'session=alice');
    assert.equal(response.headers.get('location'),'https://broker.example.test/consent');
    assert.match(response.headers.get('set-cookie'),/__Host-gms_gmail_state=synthetic/);
  } finally {globalThis.fetch=original;delete process.env.GMS_PUBLIC_ORIGIN;delete process.env.GMS_FULL_WORKER_ROUTES;}
});
test('Gmail consent callback accepts cross-site navigation but keeps account cookies bound',async()=>{
  const callback=await load('../app/api/auth/gmail-callback/route.ts');
  process.env.GMS_PUBLIC_ORIGIN=origin;
  process.env.GMS_FULL_WORKER_ROUTES='1';
  let call;
  globalThis.fetch=async(url,init)=>{
    call={url:String(url),init};
    return new Response(null,{status:303,headers:{location:'/settings','Set-Cookie':'__Host-gms_gmail_state=; Max-Age=0; Secure; HttpOnly; Path=/'}});
  };
  try {
    const req=new Request(origin+'/api/auth/gmail-callback?gmail_consent=synthetic',{headers:{host:'gms.example.test',cookie:'session=alice; state=bound','sec-fetch-site':'cross-site'}});
    req.nextUrl=new URL(req.url);
    const response=await callback.GET(req);
    assert.equal(response.status,303);
    assert.equal(call.url,'http://backend.test/api/auth/gmail-callback?gmail_consent=synthetic');
    assert.equal(call.init.redirect,'manual');
    assert.equal(call.init.headers.cookie,'session=alice; state=bound');
    assert.equal(response.headers.get('referrer-policy'),'no-referrer');
    assert.match(response.headers.get('set-cookie'),/Max-Age=0/);
  } finally {globalThis.fetch=original;delete process.env.GMS_PUBLIC_ORIGIN;delete process.env.GMS_FULL_WORKER_ROUTES;}
});
