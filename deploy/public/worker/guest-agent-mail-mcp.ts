// Fixed full-agent extension. No environment or workspace routing config.
import { createMcpAdapter } from './pi-pkgs/node_modules/pi-mcp-adapter/index.ts';
import { MAIL_SERVER, mailAdapterConfig } from './guest-mail-server.ts';


export default createMcpAdapter({ config: mailAdapterConfig(MAIL_SERVER) });
