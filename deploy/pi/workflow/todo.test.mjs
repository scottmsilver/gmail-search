import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire, stripTypeScriptTypes } from 'node:module';
import { pathToFileURL } from 'node:url';
import test from 'node:test';

const require = createRequire(pathToFileURL(`${process.env.PI_PKGS_ROOT || new URL('../pi-pkgs', import.meta.url).pathname}/package.json`));
const source = stripTypeScriptTypes(await readFile(new URL('./todo.ts', import.meta.url), 'utf8'))
  .replace('"typebox"', JSON.stringify(pathToFileURL(require.resolve('typebox')).href));
const { default: install } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
function harness() {
  const events = new Map();
  let tool;
  install({ on: (name, fn) => events.set(name, fn), registerTool: (value) => { tool = value; } });
  return {
    call: (params) => tool.execute('test', params),
    restore: (event, results) => events.get(event)({}, { sessionManager: {
      getBranch: () => results.map((result) => ({ type: 'message', message: {
        role: 'toolResult', toolName: 'todo', ...result,
      } })),
    } }),
  };
}
test('add/update/list/clear preserve IDs and immutable snapshots', async () => {
  const h = harness();
  const first = await h.call({ action: 'add', text: 'Read source' });
  assert.deepEqual(first.details.todos, [{ id: 1, text: 'Read source', status: 'pending' }]);
  await h.call({ action: 'update', id: 1, status: 'in_progress', text: 'Read thread' });
  assert.equal(first.details.todos[0].status, 'pending');
  assert.equal((await h.call({ action: 'list' })).details.todos[0].text, 'Read thread');
  await h.call({ action: 'update', id: 1, status: 'completed' });
  await h.call({ action: 'clear' });
  assert.equal((await h.call({ action: 'add', text: 'Check date' })).details.todos[0].id, 2);
});
for (const event of ['session_start', 'session_switch', 'session_fork', 'session_tree']) {
  test(`${event} restores the selected branch and an empty session resets state`, async () => {
    const h = harness();
    const first = await h.call({ action: 'add', text: 'Source A' });
    await h.call({ action: 'add', text: 'Source B' });
    await h.restore(event, [first]);
    const next = await h.call({ action: 'add', text: 'Source C' });
    assert.deepEqual(next.details.todos.map((t) => t.text), ['Source A', 'Source C']);
    assert.equal(next.details.nextId, 3);
    await h.restore(event, []);
    assert.deepEqual((await h.call({ action: 'list' })).details.todos, []);
  });
}
test('invalid requests return errors without changing state', async () => {
  const h = harness();
  await h.call({ action: 'add', text: 'A' });
  for (const params of [{ action: 'add', text: ' ' }, { action: 'update', id: 99, status: 'completed' },
    { action: 'update', id: 1 }, { action: 'update', id: 1, status: 'bogus' }]) {
    const result = await h.call(params);
    assert.equal(result.isError, true);
    assert.deepEqual(result.details.todos, [{ id: 1, text: 'A', status: 'pending' }]);
  }
});
