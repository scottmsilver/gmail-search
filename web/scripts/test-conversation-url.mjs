import assert from 'node:assert/strict';
import {test} from 'node:test';
import {tsImport} from 'tsx/esm/api';
const mod = await tsImport(new URL('../lib/conversationUrl.ts', import.meta.url).pathname, import.meta.url);
const {conversationPath, conversationIdFromPath, legacyConversationUrl} = mod.default ?? mod;

test('a conversation lives at /c/<id>', () => {
 assert.equal(conversationPath('a1b2c3d4e5f6'), '/c/a1b2c3d4e5f6');
 assert.equal(conversationIdFromPath('/c/a1b2c3d4e5f6'), 'a1b2c3d4e5f6');
 for (const path of ['/', '/search', '/c/', '/c/abc', '/c/a1b2c3d4e5f6/x', '/c/../etc', null]) assert.equal(conversationIdFromPath(path), null);
});

test('old /?c=<id> links move to /c/<id>, keeping the rest of the query', () => {
 assert.equal(legacyConversationUrl(new URLSearchParams('c=a1b2c3d4e5f6')), '/c/a1b2c3d4e5f6');
 assert.equal(legacyConversationUrl(new URLSearchParams('c=a1b2c3d4e5f6&thread=18f3abc')), '/c/a1b2c3d4e5f6?thread=18f3abc');
 for (const q of ['', 'thread=18f3abc', 'c=../x', 'c=ab']) assert.equal(legacyConversationUrl(new URLSearchParams(q)), null);
});
