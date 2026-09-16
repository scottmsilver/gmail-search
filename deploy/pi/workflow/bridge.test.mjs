import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createJiti } from '../pi-pkgs/node_modules/jiti/lib/jiti.mjs';
const jiti = createJiti(import.meta.url);
const { installWorkflowBridge, LIVENESS_KEY } = await jiti.import('./bridge.ts');
const { default: telemetry } = await jiti.import('./telemetry.ts');
const { registerPiWebSessionLiveness } = await jiti.import('../pi-pkgs/node_modules/pi-subagents/src/integrations/pi-web-session-liveness.ts');

function fixture() {
  const hooks = new Map(), commands = new Map(), listeners = new Map(), output = [];
  const pi = {
    on(name, fn) { hooks.set(name, [...(hooks.get(name) || []), fn]); },
    registerCommand(name, command) { commands.set(name, command); },
    events: {
      on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); },
      emit(name, payload) { listeners.get(name)?.(payload); },
    },
  };
  const ctx = { sessionManager: { getSessionId: () => 'parent', getSessionFile: () => '/parent.jsonl' }, isIdle: () => true };
  return { pi, ctx, hooks, commands, output, fire: async (name, event = {}, context = ctx) => { for (const fn of hooks.get(name) || []) await fn(event, context); } };
}

test('registry scopes providers, tracks pending delivery, and disposes replacement safely', async () => {
  const f = fixture();
  installWorkflowBridge(f.pi, { write: value => f.output.push(value) });
  const registry = globalThis[Symbol.for(LIVENESS_KEY)];
  await f.commands.get('gms-workflow-status').handler('', f.ctx);
  assert.equal(f.output.at(-1).ready, false);
  let pending = true;
  const old = registry.register({ name: 'pi-subagents', sessionId: 'parent', isActive: () => true });
  registry.register({ name: 'pi-subagents', sessionId: 'parent', isActive: () => pending });
  registry.register({ name: 'unrelated', sessionId: 'other', isActive: () => true });
  old();
  await f.commands.get('gms-workflow-status').handler('', f.ctx);
  assert.deepEqual(f.output.at(-1), { type: 'gms_workflow_state', ready: true, active: true });
  pending = false;
  await f.commands.get('gms-workflow-status').handler('', f.ctx);
  assert.equal(f.output.at(-1).active, false);
  f.ctx.hasPendingMessages = () => true;
  await f.commands.get('gms-workflow-status').handler('', f.ctx);
  assert.equal(f.output.at(-1).active, true);
});

test('stop targets only current session status runs and waits for liveness', async () => {
  const f = fixture(); let active = true; const stops = [];
  installWorkflowBridge(f.pi, { write: value => f.output.push(value), stopTimeoutMs: 100 });
  globalThis[Symbol.for(LIVENESS_KEY)].register({name:'pi-subagents', sessionId:'parent', isActive:()=>active});
  f.pi.events.on('subagents:rpc:v1:request', request => {
    if (request.method === 'stop') { stops.push(request.params.id); active = false; }
    f.pi.events.emit(`subagents:rpc:v1:reply:${request.requestId}`, { success:true, data: request.method === 'status' ? {asyncSnapshot:{runs:[{id:'owned-run',state:'running'}, {id:'done',state:'complete'}]}} : {} });
  });
  await f.commands.get('gms-workflow-stop').handler('', f.ctx);
  assert.deepEqual(stops, ['owned-run']);
  assert.equal(f.output.at(-1).active, false);
});

test('installed pi-subagents accepts the host registry and releases its provider', async () => {
  const f = fixture(); const bridge = installWorkflowBridge(f.pi);
  const handle = registerPiWebSessionLiveness({sessionId:'parent',sessionFile:'/parent.jsonl',isActive:()=>true});
  assert.equal(handle.registered,true);
  assert.equal(bridge.snapshot(f.ctx).ready,true);
  assert.equal(bridge.snapshot(f.ctx).active,true);
  handle.release();
  assert.equal(bridge.snapshot(f.ctx).ready,false);
});

test('stop reports incomplete cleanup instead of claiming children stopped', async () => {
  const f = fixture(); installWorkflowBridge(f.pi,{write:value=>f.output.push(value),stopTimeoutMs:20});
  globalThis[Symbol.for(LIVENESS_KEY)].register({name:'pi-subagents',sessionId:'parent',isActive:()=>true});
  f.pi.events.on('subagents:rpc:v1:request', request => {
    f.pi.events.emit(`subagents:rpc:v1:reply:${request.requestId}`,{success:true,data:{asyncSnapshot:{runs:[]}}});
  });
  await f.commands.get('gms-workflow-stop').handler('',f.ctx);
  assert.equal(f.output.at(-1).active,true);
  assert.match(f.output.at(-1).error,/did not complete/);
});

test('telemetry records child usage and tool summaries without raw tool bodies', async () => {
  const f = fixture(); const directory = mkdtempSync(join(tmpdir(), 'gms-bridge-'));
  const previous = {trace:process.env.GMS_WORKFLOW_TRACE, root:process.env.GMS_WORKFLOW_ROOT_SESSION};
  process.env.GMS_WORKFLOW_TRACE = join(directory,'events.jsonl');
  process.env.GMS_WORKFLOW_ROOT_SESSION = 'root';
  try {
    telemetry(f.pi);
    await f.fire('session_start');
    const message = {role:'assistant',id:'m1',model:'model',provider:'provider',usage:{input:7,output:3,cacheRead:2,cacheWrite:0,cost:{total:0.01}},content:[{type:'text',text:'private answer'}]};
    await f.fire('message_end',{message}); await f.fire('message_end',{message});
    await f.fire('tool_execution_start',{toolCallId:'t1',toolName:'gmail_search',args:{query:'bounded query'}});
    await f.fire('tool_execution_end',{toolCallId:'t1',toolName:'gmail_search',isError:true,result:{content:[{type:'text',text:'PRIVATE EMAIL BODY'}]}});
    const text = readFileSync(process.env.GMS_WORKFLOW_TRACE,'utf8'); const rows = text.trim().split('\n').map(JSON.parse);
    assert.equal(rows.filter(r=>r.type==='message_end').length,1);
    assert.equal(rows[1].parent,false); assert.equal(rows[1].usage.cost.total,0.01);
    assert.equal(rows.at(-1).is_error,true); assert.ok(rows.at(-1).result_bytes > 0);
    assert.equal(text.includes('PRIVATE EMAIL BODY'),false); assert.equal(text.includes('private answer'),false);
    await f.fire('session_shutdown');
    assert.equal(JSON.parse(readFileSync(process.env.GMS_WORKFLOW_TRACE,'utf8').trim().split('\n').at(-1)).type,'session_end');
  } finally {
    for (const [key,value] of [['GMS_WORKFLOW_TRACE',previous.trace],['GMS_WORKFLOW_ROOT_SESSION',previous.root]]) { if (value === undefined) delete process.env[key]; else process.env[key]=value; }
    rmSync(directory,{recursive:true,force:true});
  }
});

test('telemetry records compaction usage and flags failed compaction accounting', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'workflow-usage-'));
  const prior = process.env.GMS_WORKFLOW_TRACE;
  process.env.GMS_WORKFLOW_TRACE = join(dir, 'trace.jsonl');
  try {
    const f = fixture();
    f.ctx.model = {provider:'test',id:'model'};
    telemetry(f.pi);
    await f.fire('session_compact', {compactionEntry:{id:'sum1',usage:{input:3,output:2,cost:{total:.2}}}});
    await f.fire('session_compact', {compactionEntry:{id:'sum1',usage:{input:3,output:2,cost:{total:.2}}}});
    await f.fire('session_compact_failed', {aborted:true});
    const rows = readFileSync(process.env.GMS_WORKFLOW_TRACE,'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(rows.filter(r=>r.type === 'summary_usage').length,1);
    assert.equal(rows.find(r=>r.type === 'summary_usage').usage.input,3);
    assert.equal(rows.filter(r=>r.type === 'accounting_gap').length,1);
  } finally {
    if (prior === undefined) delete process.env.GMS_WORKFLOW_TRACE; else process.env.GMS_WORKFLOW_TRACE = prior;
    rmSync(dir,{recursive:true,force:true});
  }
});
