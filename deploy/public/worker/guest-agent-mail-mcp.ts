// Fixed full-agent extension. No environment or workspace routing config.
import { createMcpAdapter } from './pi-pkgs/node_modules/pi-mcp-adapter/index.ts';

const tools = ['describe_schema', 'sql_query_batch', 'get_thread_batch', 'publish_artifact_batch',
  'search_emails_batch', 'find_facts', 'query_emails_batch', 'get_attachment_batch'];

export default createMcpAdapter({
  config: {
    imports: [],
    settings: {
      hostConfigDiscovery: 'off', agentPluginPaths: [],
      directTools: true, disableProxyTool: true, scriptMode: false,
      autoAuth: false, sampling: false, elicitation: false, outputGuard: { maxBytes: 8 * 1024 * 1024, maxLines: 100000, detailsMaxBytes: 16 * 1024 },
      warnOnLargeDirectTools: false, notifyOnStartupConnect: false,
      toolPrefix: 'server', requestTimeoutMs: 30000,
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
