// The guest's one MCP server, shared by the mail extensions (which run it) and
// the workflow extension (which describes it to pi-subagents for children).
export const MAIL_TOOLS = ['describe_schema', 'sql_query_batch', 'get_thread_batch', 'publish_artifact_batch',
  'search_emails_batch', 'find_facts', 'query_emails_batch', 'get_attachment_batch', 'judge'];

export const MAIL_SERVER = {
  command: '/usr/bin/env',
  args: ['-i', 'PATH=/usr/bin:/bin', 'LANG=C.UTF-8', '/usr/bin/python3', '-I', '/tmp/runtime/guest_mail_mcp.py'],
  lifecycle: 'lazy-keep-alive', idleTimeout: 0,
  directTools: MAIL_TOOLS, includeTools: MAIL_TOOLS, exposeResources: false,
  auth: false, oauth: false, protocolVersion: 'legacy',
};

// A parallel mail-researcher's copy: the server applies its mail call budget.
// Foreground children run inside the parent's Pi process, so this flag is the
// only thing that tells their server apart from the parent's.
export const SUBAGENT_MAIL_SERVER = { ...MAIL_SERVER, args: [...MAIL_SERVER.args, '--subagent'] };

// outputGuard bounds what the model sees per tool call (~25k tokens, Claude
// Code's own MCP default). Larger results keep a preview plus a notice naming a
// temp file holding the full output, which Pi's read/grep tools can page. At
// 8 MiB one 694 KB thread fetch filled Gemini's 200k-token window.
// requestTimeoutMs is a backstop just above the 5 s tool deadline that
// guest_mail_tools.py and the gateway enforce, so their clearer error wins.
export function mailAdapterConfig(server: typeof MAIL_SERVER) {
  return {
    imports: [],
    settings: {
      hostConfigDiscovery: 'off', agentPluginPaths: [],
      directTools: true, disableProxyTool: true, scriptMode: false,
      autoAuth: false, sampling: false, elicitation: false, outputGuard: { maxBytes: 100 * 1024, maxLines: 5000, detailsMaxBytes: 16 * 1024 },
      warnOnLargeDirectTools: false, notifyOnStartupConnect: false,
      toolPrefix: 'server', requestTimeoutMs: 6000,
    },
    mcpServers: { mail: server },
  };
}
