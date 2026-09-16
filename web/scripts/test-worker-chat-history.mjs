import assert from 'node:assert/strict';
import {test} from 'node:test';
import {tsImport} from 'tsx/esm/api';
const imported = await tsImport(new URL('../lib/workerChatHistory.ts', import.meta.url).pathname, import.meta.url);
const {workerChatHistory} = imported.default ?? imported;
test('follow-up request keeps user text without round-tripping rich tool results', () => {
  const history = [{id:'u1',role:'user',parts:[{type:'text',text:'Find receipts'}]},
    {id:'a1',role:'assistant',parts:[{type:'data-deep-stage',data:{output:'a'.repeat(300000)}},{type:'text',text:'Found receipts'}]},
    {id:'u2',role:'user',parts:[{type:'text',text:'Total them'}, {type:'data-extra',data:'untrusted'}]}];
  assert.deepEqual(workerChatHistory(history), [history[0], {id:'u2',role:'user',parts:[{type:'text',text:'Total them'}]}]);
  assert.ok(JSON.stringify(workerChatHistory(history)).length < 1024);
  assert.equal(history[1].parts[0].data.output.length, 300000);
});
