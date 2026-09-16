import type {UIMessage} from 'ai';

/** Browser submissions contain user text; rich assistant history stays server-owned. */
export function workerChatHistory(messages: UIMessage[]): UIMessage[] {
  return messages.filter(message => message.role === 'user').map(message => ({
    id: message.id,
    role: 'user' as const,
    parts: message.parts.filter(part => part.type === 'text').map(part => ({type: 'text' as const, text: part.text})),
  }));
}
