import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { convertToModelMessages, stepCountIs, streamText, type UIMessage } from "ai";
import { NextRequest } from "next/server";

import { AGENT_MODEL, geminiApiKey } from "@/lib/config";
import { buildSystemPrompt } from "@/lib/systemPrompt";
import { buildTools } from "@/lib/tools";

export const runtime = "nodejs";
export const maxDuration = 120;

const MAX_TOOL_STEPS = 6;

export async function POST(req: NextRequest) {
  const { messages } = (await req.json()) as { messages: UIMessage[] };

  if (!Array.isArray(messages) || messages.length === 0) {
    return new Response(JSON.stringify({ error: "messages required" }), { status: 400 });
  }

  const google = createGoogleGenerativeAI({ apiKey: geminiApiKey() });
  const tools = buildTools();
  const system = buildSystemPrompt(tools);

  const result = streamText({
    model: google(AGENT_MODEL),
    system,
    messages: await convertToModelMessages(messages),
    tools,
    stopWhen: stepCountIs(MAX_TOOL_STEPS),
  });

  return result.toUIMessageStreamResponse();
}
