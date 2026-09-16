import { createMcpAdapter } from 'pi-mcp-adapter';

export default createMcpAdapter({ configPath:process.env.GMS_MCP_CONFIG || '/opt/gmail-mcp.json' });
