import assert from 'node:assert/strict';
import {mkdtempSync, writeFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';
import {tsImport} from 'tsx/esm/api';

const load = async (rel) => { const mod = await tsImport(new URL(rel, import.meta.url).pathname, import.meta.url); return mod.default ?? mod; };
const notes = await load('../lib/whatsNew.ts');
const route = await load('../app/api/whats-new/route.ts');

const good = {releases: [
 {release: 'web-20260925', target: 'abc1234', issues: [{number: 7, pr: 8, title: 'feat(web): <b>plain</b> text'}]},
 {release: 'old-20260920', target: 'def5678', issues: [{number: 3, pr: null, title: 'fix: older'}]},
]};

test('a well-formed document passes through unchanged', () => {
 assert.deepEqual(notes.sanitizeWhatsNew(good), good);
 assert.equal(notes.latestRelease(good), 'web-20260925');
});

test('malformed documents are no releases and malformed entries are dropped', () => {
 for (const bad of [null, 'x', [], {releases: 5}, {}]) assert.deepEqual(notes.sanitizeWhatsNew(bad), {releases: []});
 for (const text of ['not json', '', 'x'.repeat(notes.WHATS_NEW_MAX_BYTES + 1)]) assert.deepEqual(notes.parseWhatsNew(text), {releases: []});
 const mixed = {releases: [null, {release: 1, issues: []}, {release: 'r', issues: [null, {number: '1', title: 't'}, {number: 2, title: 't', pr: 'x'}]}]};
 assert.deepEqual(notes.sanitizeWhatsNew(mixed), {releases: [{release: 'r', target: '', issues: [{number: 2, pr: null, title: 't'}]}]});
 assert.equal(notes.latestRelease({releases: []}), null);
});

test('the popup fires only for a newer release than one this browser recorded', () => {
 assert.equal(notes.isFreshRelease('b', 'a'), true);
 assert.equal(notes.isFreshRelease('a', 'a'), false);
 assert.equal(notes.isFreshRelease('a', null), false, 'a first-ever visit does not pop');
 assert.equal(notes.isFreshRelease(null, 'a'), false, 'no notes, nothing to show');
});

test('storage that throws reads as a first visit and never breaks the page', () => {
 const throwing = {getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); }};
 assert.equal(notes.readSeenRelease(throwing), null);
 assert.doesNotThrow(() => notes.writeSeenRelease(throwing, 'r'));
 assert.equal(notes.readSeenRelease(undefined), null);
 const store = new Map();
 const memory = {getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v)};
 notes.writeSeenRelease(memory, 'r1');
 assert.equal(notes.readSeenRelease(memory), 'r1');
});

test('the client fetch degrades to no notes on a failed or broken response', async () => {
 assert.deepEqual(await notes.fetchWhatsNew(async () => new Response('nope', {status: 404})), {releases: []});
 assert.deepEqual(await notes.fetchWhatsNew(async () => { throw new Error('offline'); }), {releases: []});
 assert.deepEqual(await notes.fetchWhatsNew(async () => new Response('{bad')), {releases: []});
 assert.deepEqual(await notes.fetchWhatsNew(async () => Response.json(good)), good);
 const huge = new Response('x'.repeat(notes.WHATS_NEW_MAX_BYTES + 1));
 assert.deepEqual(await notes.fetchWhatsNew(async () => huge), {releases: []});
});

const origin = 'https://gms.example.test';
const request = () => new Request(origin + '/api/whats-new', {headers: {host: 'gms.example.test'}});

async function withRelease(text, fn) {
 const dir = mkdtempSync(join(tmpdir(), 'whats-new-'));
 const cwd = process.cwd(), fetchImpl = globalThis.fetch;
 if (text !== null) writeFileSync(join(dir, 'whats-new.json'), text);
 process.chdir(dir);
 process.env.GMS_PUBLIC_ORIGIN = origin;
 process.env.PYTHON_API_URL = 'http://backend.test';
 globalThis.fetch = async () => Response.json({multi_tenant: true, user: {id: 'u1'}});
 try { await fn(); } finally {
  process.chdir(cwd); globalThis.fetch = fetchImpl; delete process.env.GMS_PUBLIC_ORIGIN; rmSync(dir, {recursive: true});
 }
}

test('the route serves the release file, and anything wrong with it is no notes', async () => {
 await withRelease(JSON.stringify(good), async () => assert.deepEqual(await (await route.GET(request(), {})).json(), good));
 for (const text of [null, 'not json', 'x'.repeat(notes.WHATS_NEW_MAX_BYTES + 1)]) {
  await withRelease(text, async () => {
   const response = await route.GET(request(), {});
   assert.equal(response.status, 200);
   assert.deepEqual(await response.json(), {releases: []});
  });
 }
});

test('the route needs a signed-in user on the public web and is absent on the owner web', async () => {
 await withRelease(JSON.stringify(good), async () => {
  globalThis.fetch = async () => Response.json({multi_tenant: true, user: null});
  assert.equal((await route.GET(request(), {})).status, 401);
  delete process.env.GMS_PUBLIC_ORIGIN;
  assert.equal((await route.GET(request(), {})).status, 404);
 });
});
