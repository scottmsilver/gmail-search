// Subagents for questions with independent parts: the parent delegates each
// part to a mail-researcher child (pi-subagents, pinned in pi-pkgs) and
// combines their answers. Children inherit the run's environment, so they use
// the same gateway, capabilities and budget as the parent.
//
// pi-subagents grants a child `mcp:mail/<tool>` only when it can resolve the
// server (agent-dir mcp.json) and a fresh tool-metadata cache whose hash
// matches that definition. Our server starts lazily, so the cache would not
// exist yet when the parent first delegates; write both here, hashed with
// pi-subagents' own function.
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import subagents from './pi-pkgs/node_modules/pi-subagents/index.ts';
import { computeMcpServerHash } from './pi-pkgs/node_modules/pi-subagents/src/runs/shared/mcp-direct-tool-allowlist.ts';
import { MAIL_SERVER, MAIL_TOOLS } from './guest-mail-server.ts';

function describeMailServerForChildren(agentDir: string) {
  writeFileSync(join(agentDir, 'mcp.json'), JSON.stringify({
    mcpServers: { mail: MAIL_SERVER }, settings: { toolPrefix: 'server', directTools: true },
  }), { mode: 0o600 });
  writeFileSync(join(agentDir, 'mcp-cache.json'), JSON.stringify({
    version: 1,
    servers: { mail: { configHash: computeMcpServerHash(MAIL_SERVER as never), cachedAt: Date.now(),
      tools: MAIL_TOOLS.map((name) => ({ name })) } },
  }), { mode: 0o600 });
}

export default async function workflow(pi: unknown) {
  const agentDir = process.env.PI_CODING_AGENT_DIR;
  if (agentDir) describeMailServerForChildren(agentDir);
  await (subagents as (api: unknown) => Promise<void>)(pi);
}
