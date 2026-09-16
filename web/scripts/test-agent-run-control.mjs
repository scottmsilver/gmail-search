import assert from 'node:assert/strict';
import {test} from 'node:test';
import {tsImport} from 'tsx/esm/api';
const mod = await tsImport(new URL('../lib/agentRunControl.ts',import.meta.url).pathname,import.meta.url);
const {stopAgentRun} = mod.default ?? mod;
const original=globalThis.fetch;
test('Stop only confirms terminal states, not pending cleanup',async()=>{
  try {
    for (const state of ['completed','cancelled','failed']) {
      globalThis.fetch=async()=>Response.json({state});
      await stopAgentRun({runId:'run',conversationId:'conversation'});
    }
    globalThis.fetch=async()=>Response.json({state:'stopping'},{status:202});
    await assert.rejects(stopAgentRun({runId:'run',conversationId:'conversation'}),/not confirmed/);
    globalThis.fetch=async()=>Response.json({detail:'unavailable'},{status:503});
    await assert.rejects(stopAgentRun({runId:'run',conversationId:'conversation'}),/not confirmed/);
  } finally {globalThis.fetch=original;}
});
