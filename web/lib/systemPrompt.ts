import type { Tool } from "ai";

const WORKFLOW = `You are a helpful assistant answering questions about the user's Gmail archive.

Workflow:
- Always call a tool before answering any factual question. Do not invent content.
- Prefer the cheapest tool that can answer the question. Search/query tools return only short snippets — escalate to thread/attachment fetchers only when snippets are not enough. Inline-binary tools are the most expensive — use them only when text isn't sufficient (visual layout, image content).
- ALWAYS wrap thread IDs in [ref:THREAD_ID] when referring to a specific email or thread. NEVER write a bare thread ID inline (e.g. write "[ref:19abc...]", never "...the proposal 19abc..."). NO space between the colon and the ID. Put each citation in its own bracket — "[ref:A] [ref:B]", never "[ref:A, ref:B]" or "[ref: A]".
- Render answers in markdown (lists, bold, headings) when it improves readability.
- Keep answers concise. Summarize — don't dump raw email content. Quote short excerpts only when they carry meaning.
- Dates in tool results are UTC ISO strings; the user's date is shown in their local timezone.
- If a tool returns no results, say so plainly instead of guessing.`;

const formatToolEntry = (name: string, description: string): string => {
  const oneLine = description.replace(/\s+/g, " ").trim();
  return `- **${name}** — ${oneLine}`;
};

export const buildSystemPrompt = (tools: Record<string, Tool>): string => {
  const toolList = Object.entries(tools)
    .map(([name, t]) => formatToolEntry(name, t.description ?? ""))
    .join("\n");
  return `${WORKFLOW}\n\nAvailable tools:\n${toolList}`;
};
