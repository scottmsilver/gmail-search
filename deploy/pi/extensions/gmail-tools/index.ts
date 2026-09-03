import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { registerMcpTools } from "./mcp-bridge.ts";

function requiredEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`${name} is not set; the gmail-tools extension cannot start`);
  return v;
}

function transportFor(url: string, token: string | undefined) {
  const init = token ? { requestInit: { headers: { Authorization: `Bearer ${token}` } } } : undefined;
  return new StreamableHTTPClientTransport(new URL(url), init);
}

export default async function gmailTools(pi: ExtensionAPI) {
  const url = requiredEnv("GMS_MCP_URL");
  const sessionId = requiredEnv("GMS_SESSION_ID");
  const client = new Client({ name: "gmail-tools-bridge", version: "0.1.0" });
  await client.connect(transportFor(url, process.env.GMAIL_MCP_SERVICE_TOKEN));
  await registerMcpTools(pi, client, { sessionId });
  pi.on("session_shutdown", async () => {
    await client.close();
  });
}
