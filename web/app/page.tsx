"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { AssistantChatTransport, useChatRuntime } from "@assistant-ui/react-ai-sdk";

import { Thread } from "@/components/Thread";
import { ThreadDrawer } from "@/components/ThreadDrawer";
import { ThreadDrawerProvider, useThreadDrawer } from "@/components/ThreadDrawerContext";

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
  const runtime = useChatRuntime({
    transport: new AssistantChatTransport({ api: "/api/chat" }),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadDrawerProvider>
        <Thread />
        <DrawerHost />
      </ThreadDrawerProvider>
    </AssistantRuntimeProvider>
  );
}
