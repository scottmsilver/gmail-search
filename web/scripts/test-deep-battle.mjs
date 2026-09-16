import assert from 'node:assert/strict';
import { test } from 'node:test';
import { tsImport } from 'tsx/esm/api';
const load = async (p) => { const m = await tsImport(new URL(p, import.meta.url).pathname, import.meta.url); return m.default ?? m; };
const { BATTLE_VARIANTS, pickTwoRandomVariants } = await load('../lib/battleVariants.ts');
test('battle draws distinct deep backends/models, including Muse', () => {
  assert.ok(BATTLE_VARIANTS.every(v => ['claude_code', 'pi'].includes(v.backend)));
  assert.ok(BATTLE_VARIANTS.some(v => v.backend === 'pi' && v.model === 'openrouter/meta/muse-spark-1.3'));
  for (let i = 0; i < 100; i++) {
    const [a, b] = pickTwoRandomVariants();
    assert.notDeepEqual(a, b);
  }
});
const { runDeepBattleSide, battleQuestion } = await load('../lib/deepBattle.ts');
const event = (kind, payload) => `event: ${kind}\ndata: ${JSON.stringify({payload})}\n\n`;
test('deep side forwards auth and model, isolates session, parses fragmented SSE and costs', async () => {
  const updates = [];
  let request;
  const sse = 'event: session\ndata: {"session_id":"s1"}\n\n' + event('plan', {approach:'Investigate'}) + event('cost', {usd:0.2}) + event('final', {text:'Answer [ref:abc]'});
  const result = await runDeepBattleSide({url:'http://local/analyze', cookie:'session=test', question:'q', variant:{backend:'pi',model:'openrouter/meta/muse-spark-1.3'}, onUpdate: s => updates.push({...s}), fetchImpl: async (_, init) => {
    request = init;
    return new Response(new ReadableStream({start(c) { for (const part of [sse.slice(0,33),sse.slice(33,80),sse.slice(80)]) c.enqueue(new TextEncoder().encode(part)); c.close(); }}));
  }});
  assert.equal(request.headers.cookie, 'session=test');
  const body = JSON.parse(request.body);
  assert.equal(body.backend, 'pi');
  assert.equal(body.model, 'openrouter/meta/muse-spark-1.3');
  assert.equal(body.conversation_id, undefined);
  assert.equal(result.answer, 'Answer [ref:abc]');
  assert.equal(result.usd, 0.2);
  assert.equal(result.sessionId, 's1');
  assert.equal(result.running, false);
  assert.ok(updates.some(u => u.steps > 0 && u.running));
});
test('errors and premature EOF finish a side without rejecting its sibling', async () => {
  for (const response of [new Response('Denied',{status:403}),new Response(event('plan',{})),new Response(event('error',{message:'failed'}))]) {
    const result = await runDeepBattleSide({url:'http://local',cookie:'',question:'q',variant:{backend:'adk',model:'test'},onUpdate:()=>{},fetchImpl:async()=>response});
    assert.equal(result.running,false);
    assert.ok(result.error);
  }
});
test('both sides get textual history including previous battle answers', () => {
  const q = battleQuestion([{role:'user',parts:[{type:'text',text:'Earlier'}]},{role:'assistant',parts:[{type:'data-battle',data:{answer_a:'One',answer_b:'Two'}}]},{role:'user',parts:[{type:'text',text:'Compare again'}]}]);
  for (const text of ['Earlier','One','Two','Compare again']) assert.ok(q.includes(text));
});

test('Pi battle pool has exactly the three requested models', () => {
  assert.deepEqual(BATTLE_VARIANTS.filter(v => v.backend === 'pi').map(v => v.model).sort(), [
    'openrouter/meta/muse-spark-1.3', 'google/gemini-3.8-flash', 'anthropic/claude-opus-5'
  ].sort());
});
