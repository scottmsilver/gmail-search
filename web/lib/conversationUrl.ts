// Conversation URLs are `/c/<id>`. The id rule matches the conversations
// API route (`/api/conversations/<id>`) and the public route allowlist.
export const CONVERSATION_ID_RE = /^[a-zA-Z0-9_-]{6,64}$/;
export const CONVERSATION_PATH_RE = /^\/c\/[a-zA-Z0-9_-]{6,64}$/;

export const conversationPath = (id: string): string => `/c/${id}`;

/** The conversation id in a `/c/<id>` pathname, else null. */
export const conversationIdFromPath = (pathname: string | null): string | null => {
  const match = /^\/c\/([^/]+)$/.exec(pathname ?? "");
  return match && CONVERSATION_ID_RE.test(match[1]) ? match[1] : null;
};

/** Where an old `/?c=<id>&…` link now lives: `/c/<id>?…` without `c`, else null. */
export const legacyConversationUrl = (search: URLSearchParams): string | null => {
  const id = search.get("c");
  if (!id || !CONVERSATION_ID_RE.test(id)) return null;
  const rest = new URLSearchParams(search);
  rest.delete("c");
  const query = rest.toString();
  return conversationPath(id) + (query ? `?${query}` : "");
};
