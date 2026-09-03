import { Type } from "typebox";

export type JsonSchema = { type?: string; properties?: Record<string, unknown>; required?: string[]; [k: string]: unknown };
type McpContent = { type: string; text?: string };
export type McpCallResult = { content?: McpContent[]; isError?: boolean; [k: string]: unknown };
export type PiToolResult = { content: { type: "text"; text: string }[]; details: unknown; isError: boolean };

type McpTool = { name: string; description?: string; inputSchema: JsonSchema };
type McpClientLike = {
  listTools(): Promise<{ tools: McpTool[] }>;
  callTool(req: { name: string; arguments: Record<string, unknown> }, schema?: undefined, opts?: { signal?: AbortSignal }): Promise<unknown>;
};
type PiLike = { registerTool(def: unknown): void };

export function stripSessionId(schema: JsonSchema): JsonSchema {
  const out = structuredClone(schema);
  if (out.properties) delete out.properties.session_id;
  out.required = (out.required ?? []).filter((r) => r !== "session_id");
  return out;
}

export function toPiResult(res: McpCallResult): PiToolResult {
  const text = (res.content ?? []).filter((c) => c.type === "text").map((c) => c.text ?? "").join("\n");
  return { content: [{ type: "text", text }], details: res, isError: Boolean(res.isError) };
}

function toolDefinition(tool: McpTool, client: McpClientLike, sessionId: string) {
  return {
    name: tool.name,
    label: tool.name,
    description: tool.description ?? "",
    parameters: Type.Unsafe<Record<string, unknown>>(stripSessionId(tool.inputSchema)),
    async execute(_id: string, params: Record<string, unknown>, signal?: AbortSignal) {
      const res = (await client.callTool({ name: tool.name, arguments: { ...params, session_id: sessionId } }, undefined, { signal })) as McpCallResult;
      return toPiResult(res);
    },
  };
}

export async function registerMcpTools(pi: PiLike, client: McpClientLike, opts: { sessionId: string }): Promise<string[]> {
  const { tools } = await client.listTools();
  for (const tool of tools) pi.registerTool(toolDefinition(tool, client, opts.sessionId));
  return tools.map((t) => t.name);
}
