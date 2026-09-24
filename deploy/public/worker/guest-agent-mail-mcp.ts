// Fixed full-agent extension. No environment or workspace routing config.
// outputGuard bounds what the model sees per tool call (~25k tokens, Claude
// Code's own MCP default). Larger results keep a preview plus a notice naming a
// temp file holding the full output, which Pi's read/grep tools can page. At
// 8 MiB one 694 KB thread fetch filled Gemini's 200k-token window.
// requestTimeoutMs is a backstop just above the 5 s tool deadline that
// guest_mail_tools.py and the gateway enforce, so their clearer error wins.
import { createMcpAdapter } from './pi-pkgs/node_modules/pi-mcp-adapter/index.ts';

const tools = ['describe_schema', 'sql_query_batch', 'get_thread_batch', 'publish_artifact_batch',
  'search_emails_batch', 'find_facts', 'query_emails_batch', 'get_attachment_batch', 'judge'];

export default createMcpAdapter({
  config: {
    imports: [],
    settings: {
      hostConfigDiscovery: 'off', agentPluginPaths: [],
      directTools: true, disableProxyTool: true, scriptMode: false,
      autoAuth: false, sampling: false, elicitation: false, outputGuard: { maxBytes: 100 * 1024, maxLines: 5000, detailsMaxBytes: 16 * 1024 },
      warnOnLargeDirectTools: false, notifyOnStartupConnect: false,
      toolPrefix: 'server', requestTimeoutMs: 6000,
    },
    mcpServers: {
      mail: {
        command: '/usr/bin/env',
        args: ['-i', 'PATH=/usr/bin:/bin', 'LANG=C.UTF-8', '/usr/bin/python3', '-I', '/tmp/runtime/guest_mail_mcp.py'],
        lifecycle: 'lazy-keep-alive', idleTimeout: 0,
        directTools: tools, includeTools: tools, exposeResources: false,
        auth: false, oauth: false, protocolVersion: 'legacy',
      },
    },
  },
});
