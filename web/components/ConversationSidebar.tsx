"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

type Props = {
  activeId: string | null;
  onNew: () => void;
};

export const ConversationSidebar = ({ activeId, onNew }: Props) => {
  const [items, setItems] = useState<Conversation[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const res = await fetch("/api/conversations");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { conversations: Conversation[] };
      setItems(data.conversations ?? []);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    void refresh();
    const i = setInterval(refresh, 10_000);
    return () => clearInterval(i);
  }, []);

  // Refresh when active conversation changes (new messages arrived on it)
  useEffect(() => {
    if (!activeId) return;
    void refresh();
  }, [activeId]);

  const remove = async (id: string) => {
    if (!confirm("Delete this conversation?")) return;
    await fetch(`/api/conversations/${id}`, { method: "DELETE" });
    void refresh();
    if (id === activeId) onNew();
  };

  return (
    <aside className="w-64 flex-shrink-0 border-r border-neutral-200 bg-neutral-50 flex flex-col">
      <div className="p-3 border-b border-neutral-200">
        <button
          type="button"
          onClick={onNew}
          className="w-full rounded-lg bg-neutral-900 text-white text-sm py-2 hover:bg-neutral-700 transition-colors"
        >
          + New chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {err && <div className="p-3 text-xs text-red-600">{err}</div>}
        {items.length === 0 && !err && (
          <div className="p-3 text-xs text-neutral-400">No conversations yet.</div>
        )}
        {items.map((c) => {
          const active = c.id === activeId;
          return (
            <div
              key={c.id}
              className={
                active
                  ? "group flex items-center gap-1 px-2 py-2 bg-white border-l-2 border-blue-500"
                  : "group flex items-center gap-1 px-2 py-2 hover:bg-white/70 border-l-2 border-transparent"
              }
            >
              <Link
                href={`/?c=${c.id}`}
                className="flex-1 min-w-0 text-sm truncate text-neutral-800"
                title={c.title}
              >
                {c.title || "New chat"}
              </Link>
              <button
                type="button"
                onClick={() => remove(c.id)}
                className="opacity-0 group-hover:opacity-100 text-neutral-400 hover:text-red-600 text-xs px-1"
                title="Delete"
                aria-label="Delete conversation"
              >
                ✕
              </button>
            </div>
          );
        })}
      </div>
    </aside>
  );
};
