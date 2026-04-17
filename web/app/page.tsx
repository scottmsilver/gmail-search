"use client";

import { useMemo } from "react";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { AssistantChatTransport, useChatRuntime } from "@assistant-ui/react-ai-sdk";

import { Thread } from "@/components/Thread";
import { ThreadDrawer } from "@/components/ThreadDrawer";
import { ThreadDrawerProvider, useThreadDrawer } from "@/components/ThreadDrawerContext";
import { getChatSettings } from "@/lib/chatSettings";

const PYTHON_UI_URL = process.env.NEXT_PUBLIC_PYTHON_UI_URL ?? "http://127.0.0.1:8080";

const DrawerHost = () => {
  const { openThreadId, setOpenThreadId } = useThreadDrawer();
  return (
    <ThreadDrawer
      threadId={openThreadId}
      onClose={() => setOpenThreadId(null)}
      pythonBaseUrl={PYTHON_UI_URL}
    />
  );
};

export default function Page() {
  // `body` resolver runs per-request, so changing the picker takes effect
  // on the next message without rebuilding the runtime.
  const transport = useMemo(
    () =>
      new AssistantChatTransport({
        api: "/api/chat",
        body: () => {
          const s = getChatSettings();
          return {
            model: s.model,
            thinkingLevel: s.thinkingLevel,
            battle: s.battleMode,
          };
        },
      }),
    [],
  );
  const runtime = useChatRuntime({ transport });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadDrawerProvider>
        <Thread />
        <DrawerHost />
      </ThreadDrawerProvider>
    </AssistantRuntimeProvider>
  );
}
