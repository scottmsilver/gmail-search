import { describe, expect, test } from "bun:test";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { z } from "zod";
import { registerMcpTools, stripSessionId, toPiResult } from "./mcp-bridge.ts";

describe("stripSessionId", () => {
  test("removes the property and the required entry without mutating input", () => {
    const schema = {
      type: "object",
      properties: { session_id: { type: "string" }, queries: { type: "array" } },
      required: ["session_id", "queries"],
    };
    const out = stripSessionId(schema);
    expect(out.properties).toEqual({ queries: { type: "array" } });
    expect(out.required).toEqual(["queries"]);
    expect(schema.required).toEqual(["session_id", "queries"]);
  });
});

describe("toPiResult", () => {
  test("joins text blocks and carries isError", () => {
    const res = { content: [{ type: "text", text: "a" }, { type: "text", text: "b" }], isError: true };
    expect(toPiResult(res)).toEqual({ content: [{ type: "text", text: "a\nb" }], details: res, isError: true });
  });
});

async function linkedClient(): Promise<Client> {
  const server = new McpServer({ name: "fake", version: "0" });
  server.tool("echo", { session_id: z.string(), text: z.string() }, async ({ session_id, text }) => ({
    content: [{ type: "text", text: `${session_id}:${text}` }],
  }));
  const [a, b] = InMemoryTransport.createLinkedPair();
  await server.connect(a);
  const client = new Client({ name: "t", version: "0" });
  await client.connect(b);
  return client;
}

describe("registerMcpTools", () => {
  test("registers every tool, hides session_id, injects it on call", async () => {
    const registered: any[] = [];
    const pi = { registerTool: (d: any) => registered.push(d) };
    const names = await registerMcpTools(pi, await linkedClient(), { sessionId: "S1" });
    expect(names).toEqual(["echo"]);
    expect(Object.keys(registered[0].parameters.properties)).toEqual(["text"]);
    const result = await registered[0].execute("id", { text: "hi" }, new AbortController().signal);
    expect(result.content[0].text).toBe("S1:hi");
    expect(result.isError).toBe(false);
  });

  test("the injected session_id wins even if the model supplies its own", async () => {
    const registered: any[] = [];
    const pi = { registerTool: (d: any) => registered.push(d) };
    await registerMcpTools(pi, await linkedClient(), { sessionId: "REAL" });
    // session_id isn't in the exposed schema, but nothing stops a model from
    // passing it anyway (e.g. via a malformed tool call) -- it must not override.
    const result = await registered[0].execute("id", { text: "hi", session_id: "SPOOFED" }, new AbortController().signal);
    expect(result.content[0].text).toBe("REAL:hi");
  });
});
