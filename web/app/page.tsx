"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { AssistantChatTransport, useChatRuntime } from "@assistant-ui/react-ai-sdk";

import { ConversationSidebar } from "@/components/ConversationSidebar";
import { ThemeEffect } from "@/components/ThemeEffect";
import { Thread } from "@/components/Thread";
import { ThreadDrawer } from "@/components/ThreadDrawer";
import { ThreadDrawerProvider, useThreadDrawer } from "@/components/ThreadDrawerContext";
import {
  getChatSettings,
  getServerChatSettings,
  setChatSettings,
  subscribeChatSettings,
} from "@/lib/chatSettings";

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

  // Current conversation id: URL wins. Null until we mint one or pick
  // one up from the URL. Minting happens in an effect (not in useState
  // initializer) so server and client agree on the initial render.
  const [conversationId, setConversationId] = useState<string | null>(urlC ?? null);

  useEffect(() => {
    if (urlC) {
      if (urlC !== conversationId) setConversationId(urlC);
      return;
    }
    if (!conversationId) {
      const id = newConversationId();
      setConversationId(id);
      router.replace(`/?c=${id}`);
    }
  }, [urlC, conversationId, router]);

  // Transport is stable across the component's lifetime; it reads the
  // latest conversation id from a ref so a change doesn't rebuild the
  // runtime (which would wipe the thread).
  const conversationIdRef = useRef<string | null>(conversationId);
  conversationIdRef.current = conversationId;
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
            conversation_id: conversationIdRef.current,
          };
        },
      }),
    [],
  );
  const runtime = useChatRuntime({ transport });
  const runtimeRef = useRef(runtime);
  runtimeRef.current = runtime;

  // Load persisted messages when switching conversations.
  const loadedId = useRef<string | null>(null);
  useEffect(() => {
    if (!conversationId) return;
    if (loadedId.current === conversationId) return;
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
        const serverMessages = data.messages ?? [];
        if (serverMessages.length === 0) {
          runtimeRef.current.thread.reset();
          return;
        }
        // AI SDK runtime expects a MessageFormatRepository — a linked
        // list of {parentId, message} entries, not a bare UIMessage
        // array. Build the chain so each message's parent is the prior
        // one's id.
        const uiMessages = serverMessages.map((m, idx) => ({
          id: `${conversationId}-${idx}`,
          role: m.role as "user" | "assistant",
          parts: m.parts,
        }));
        const repo = {
          messages: uiMessages.map((msg, idx) => ({
            parentId: idx === 0 ? null : uiMessages[idx - 1].id,
            message: msg,
          })),
          headId: uiMessages[uiMessages.length - 1].id,
        };
        runtimeRef.current.thread.importExternalState(repo);
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
    <div className="flex h-full relative">
      <ThemeEffect />
      <SidebarToggle />
      <ConversationSidebarHost activeId={conversationId} onNew={startNew} />
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

const useSidebarOpen = () =>
  useSyncExternalStore(
    subscribeChatSettings,
    () => getChatSettings().sidebarOpen,
    () => getServerChatSettings().sidebarOpen,
  );

const ConversationSidebarHost = ({
  activeId,
  onNew,
}: {
  activeId: string | null;
  onNew: () => void;
}) => {
  const open = useSidebarOpen();
  if (!open) return null;
  return <ConversationSidebar activeId={activeId} onNew={onNew} />;
};

const SidebarToggle = () => {
  const open = useSidebarOpen();
  return (
    <button
      type="button"
      onClick={() => setChatSettings({ sidebarOpen: !open })}
      className="absolute top-3 left-3 z-20 w-11 h-11 flex items-center justify-center rounded-md transition-colors theme-hover"
      style={{ color: "var(--fg-secondary)" }}
      title={open ? "Hide conversations" : "Show conversations"}
      aria-label={open ? "Hide conversations" : "Show conversations"}
    >
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
      </svg>
    </button>
  );
};
