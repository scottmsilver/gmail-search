import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import subagents from 'pi-subagents';
import gmailMcp from './gmail-mcp.ts';
import todo from './todo.ts';
import telemetry from './telemetry.ts';
import { installWorkflowBridge } from './bridge.ts';

export default async function workflow(pi: ExtensionAPI) {
  installWorkflowBridge(pi);
  // Register first so parent telemetry and subsequently spawned children agree.
  pi.on('session_start', (_event,ctx) => {
    process.env.GMS_WORKFLOW_ROOT_SESSION = ctx.sessionManager.getSessionId();
  });
  telemetry(pi);
  gmailMcp(pi);
  todo(pi);
  await subagents(pi);
}
