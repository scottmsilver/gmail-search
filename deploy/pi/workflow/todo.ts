/** Optional, branch-local task tracking adapted from Pi's examples/extensions/todo.ts.
 * Persist snapshots in tool result details so session history owns the state.
 */
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

type Status = "pending" | "in_progress" | "completed";
interface Todo { id: number; text: string; status: Status }
interface Snapshot { version: 1; todos: Todo[]; nextId: number }
const statuses = ["pending", "in_progress", "completed"] as const;
const enumSchema = (values: readonly string[]) => Type.Union(values.map((value) => Type.Literal(value)));

export default function (pi: ExtensionAPI) {
  let todos: Todo[] = [];
  let nextId = 1;
  const snapshot = (): Snapshot => ({ version: 1, todos: todos.map((todo) => ({ ...todo })), nextId });
  const restore = (ctx: ExtensionContext) => {
    todos = [];
    nextId = 1;
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type !== "message" || entry.message.role !== "toolResult" || entry.message.toolName !== "todo") continue;
      const state = entry.message.details as Snapshot | undefined;
      if (state?.version !== 1 || !Array.isArray(state.todos) || !Number.isSafeInteger(state.nextId)) continue;
      todos = state.todos.map((todo) => ({ ...todo }));
      nextId = state.nextId;
    }
  };
  pi.on("session_start", async (_event, ctx) => restore(ctx));
  pi.on("session_switch", async (_event, ctx) => restore(ctx));
  pi.on("session_fork", async (_event, ctx) => restore(ctx));
  pi.on("session_tree", async (_event, ctx) => restore(ctx));

  pi.registerTool({
    name: "todo",
    label: "Todo",
    description: "Optional session task list for complex work. Actions: list, add (text, optional status), update (id, text and/or status), clear. No task list is required to answer a question.",
    parameters: Type.Object({
      action: enumSchema(["list", "add", "update", "clear"]),
      text: Type.Optional(Type.String({ description: "Task text for add or update" })),
      id: Type.Optional(Type.Integer({ minimum: 1, description: "Stable task ID for update" })),
      status: Type.Optional(enumSchema(statuses)),
    }),
    async execute(_toolCallId, params) {
      const result = (text: string, error = false) => ({
        content: [{ type: "text" as const, text }],
        details: { ...snapshot(), action: params.action, ...(error ? { error: text } : {}) },
        ...(error ? { isError: true } : {}),
      });
      if (params.status !== undefined && !statuses.includes(params.status as Status)) return result("Invalid task status", true);
      if (params.text !== undefined && !params.text.trim()) return result("Task text must not be empty", true);
      switch (params.action) {
        case "list":
          return result(todos.length ? todos.map((todo) => `#${todo.id} [${todo.status}] ${todo.text}`).join("\n") : "No tasks");
        case "add": {
          if (!params.text) return result("Text is required for add", true);
          const todo: Todo = { id: nextId++, text: params.text.trim(), status: (params.status as Status | undefined) ?? "pending" };
          todos.push(todo);
          return result(`Added #${todo.id}: ${todo.text}`);
        }
        case "update": {
          const todo = todos.find((candidate) => candidate.id === params.id);
          if (!todo) return result(`Task #${params.id ?? "?"} not found`, true);
          if (params.text === undefined && params.status === undefined) return result("Update requires text or status", true);
          if (params.text !== undefined) todo.text = params.text.trim();
          if (params.status !== undefined) todo.status = params.status as Status;
          return result(`#${todo.id} [${todo.status}] ${todo.text}`);
        }
        case "clear":
          todos = [];
          return result("Cleared tasks");
        default:
          return result(`Unknown action: ${params.action}`, true);
      }
    },
  });
}
