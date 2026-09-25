// The mail extension for parallel mail-researcher children: the same tools,
// served with the per-child mail call budget (see guest_mail_mcp.py).
import { createMcpAdapter } from './pi-pkgs/node_modules/pi-mcp-adapter/index.ts';
import { SUBAGENT_MAIL_SERVER, mailAdapterConfig } from './guest-mail-server.ts';


export default createMcpAdapter({ config: mailAdapterConfig(SUBAGENT_MAIL_SERVER) });
