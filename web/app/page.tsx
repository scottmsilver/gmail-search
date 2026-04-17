"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { AssistantChatTransport, useChatRuntime } from "@assistant-ui/react-ai-sdk";

import { ConversationSidebar } from "@/components/ConversationSidebar";
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

const newConversationId = () =>
  // Short url-safe ID, 12 hex chars — 48 bits, plenty for personal use.
  [...crypto.getRandomValues(new Uint8Array(6))]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

export default function Page() {
  const router = useRouter();
  const params = useSearchParams();
  const urlC = params.get("c");

  // Current conversation id: URL wins. If the URL has none, mint a new
  // one; the first message's save will create the DB row.
  const [conversationId, setConversationId] = useState<string>(() => urlC || newConversationId());

  // Keep state synced to URL.
  useEffect(() => {
    if (urlC && urlC !== conversationId) setConversationId(urlC);
  }, [urlC, conversationId]);

  // When we mint a fresh id locally, push it to the URL so reload /
  // share keeps working.
  useEffect(() => {
    if (!urlC && conversationId) {
      router.replace(`/?c=${conversationId}`);
    }
  }, [urlC, conversationId, router]);

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
            conversation_id: conversationId,
          };
        },
      }),
    [conversationId],
  );
  const runtime = useChatRuntime({ transport });
  const runtimeRef = useRef(runtime);
  runtimeRef.current = runtime;

  // Load persisted messages when switching conversations.
  const loadedId = useRef<string | null>(null);
  useEffect(() => {
    if (!conversationId || loadedId.current === conversationId) return;
    loadedId.current = conversationId;
    void (async () => {
      try {
        const res = await fetch(`/api/conversations/${conversationId}`);
        if (!res.ok) {
          // New conversation — reset to empty.
          runtimeRef.current.thread.reset();
          return;
        }
        const data = (await res.json()) as {
          messages: Array<{ seq: number; role: string; parts: unknown[] }>;
        };
        // Convert our stored shape into the AI SDK UIMessage format that
        // assistant-ui's AI SDK runtime expects.
        const uiMessages = data.messages.map((m, idx) => ({
          id: `${conversationId}-${idx}`,
          role: m.role as "user" | "assistant",
          parts: m.parts,
        }));
        runtimeRef.current.thread.importExternalState(uiMessages);
      } catch (err) {
        console.error("load conversation failed", err);
      }
    })();
  }, [conversationId]);

  const startNew = useCallback(() => {
    const id = newConversationId();
    setConversationId(id);
    loadedId.current = null;
    runtimeRef.current.thread.reset();
    router.replace(`/?c=${id}`);
  }, [router]);

  return (
    <div className="flex h-screen">
      <ConversationSidebar activeId={conversationId} onNew={startNew} />
      <div className="flex-1 min-w-0">
        <AssistantRuntimeProvider runtime={runtime}>
          <ThreadDrawerProvider>
            <Thread />
            <DrawerHost />
          </ThreadDrawerProvider>
        </AssistantRuntimeProvider>
      </div>
    </div>
  );
}
