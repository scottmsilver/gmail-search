import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';

export default function telemetry(pi: ExtensionAPI) {
  let sequence = 0;
  const seen = new Set<string>();
  const started = new Map<string, number>();
  const bytes = (value: unknown) => Buffer.byteLength(JSON.stringify(value) ?? '');
  function record(type: string, ctx: ExtensionContext, data: Record<string, unknown> = {}) {
    const path = process.env.GMS_WORKFLOW_TRACE;
    if (!path) return;
    const sessionId = ctx.sessionManager.getSessionId();
    const row = { type, timestamp:new Date().toISOString(), pi_session_id:sessionId,
      parent:sessionId === process.env.GMS_WORKFLOW_ROOT_SESSION,
      event_id:`${sessionId}:${process.pid}:${++sequence}`, ...data };
    try {
      mkdirSync(dirname(path), { recursive:true });
      appendFileSync(path, `${JSON.stringify(row)}\n`, {encoding:'utf8',mode:0o600});
    } catch { console.error('Gmail workflow telemetry could not be written'); }
  }
  pi.on('session_start', (_event,ctx) => record('session_start',ctx));
  pi.on('session_shutdown', (_event,ctx) => record('session_end',ctx));
  // Summary calls are billed separately from assistant messages. The pinned Pi
  // API exposes their usage on compaction/tree entries; never count tool-result
  // usage, which may project already-recorded child spend into the parent.
  function summary(entry: any, ctx: ExtensionContext) {
    if (!entry) return;
    const key = `summary:${ctx.sessionManager.getSessionId()}:${entry.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    if (!entry.usage) {
      record('accounting_gap',ctx,{reason:'summary_usage_unavailable',entry_id:entry.id});
      return;
    }
    record('summary_usage',ctx,{entry_id:entry.id,model:ctx.model?.id,
      provider:ctx.model?.provider,usage:entry.usage});
  }
  pi.on('session_compact', (event,ctx) => summary(event.compactionEntry,ctx));
  pi.on('session_tree', (event,ctx) => summary(event.summaryEntry,ctx));
  pi.on('session_compact_failed', (_event,ctx) =>
    record('accounting_gap',ctx,{reason:'failed_or_cancelled_compaction_usage_unavailable'}));
  pi.on('message_end', (event,ctx) => {
    const message = event.message as any;
    if (message.role !== 'assistant') return;
    // Pi messages may omit an id; those retain unique per-session event IDs.
    const id = message.id ?? message.messageId;
    const key = id ? `${ctx.sessionManager.getSessionId()}:${id}` : undefined;
    if (key && seen.has(key)) return;
    if (key) seen.add(key);
    record('message_end',ctx,{ message_id:id,model:message.model,provider:message.provider,
      usage:message.usage,stop_reason:message.stopReason,is_error:message.stopReason === 'error' });
  });
  pi.on('tool_execution_start', (event,ctx) => {
    started.set(event.toolCallId,Date.now());
    // Summarize arbitrary arguments rather than persisting mail bodies or credentials.
    const args = event.args as Record<string,unknown> | undefined;
    const identifiers = Object.fromEntries(Object.entries(args ?? {}).filter(([key,value]) =>
      /^(id|thread_id|message_id|threadId|messageId|server|tool|name)$/.test(key) && typeof value === 'string'
    ).map(([key,value]) => [key,(value as string).slice(0,256)]));
    record('tool_start',ctx,{tool_call_id:event.toolCallId,tool_name:event.toolName,
      argument_keys:Object.keys(args ?? {}),argument_bytes:bytes(event.args),identifiers});
  });
  pi.on('tool_execution_end', (event,ctx) => {
    const start = started.get(event.toolCallId); started.delete(event.toolCallId);
    record('tool_end',ctx,{tool_call_id:event.toolCallId,tool_name:event.toolName,
      elapsed_ms:start === undefined ? undefined : Date.now()-start,
      is_error:event.isError,result_bytes:bytes(event.result)});
  });
}
