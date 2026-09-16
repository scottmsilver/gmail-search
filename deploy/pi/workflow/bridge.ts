import { randomUUID } from 'node:crypto';
import { writeSync } from 'node:fs';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';

// Versioned host contract implemented by pi-subagents 0.67.0's
// src/integrations/pi-web-session-liveness.ts. Includes completion batching.
export const LIVENESS_KEY = '@agegr/pi-web/session-liveness/v1';
interface Provider { name: string; sessionId: string; sessionFile?: string; isActive(): boolean }
interface Options { write?: (value: Record<string, unknown>) => void; stopTimeoutMs?: number }

export function installWorkflowBridge(pi: ExtensionAPI, options: Options = {}) {
  const providers = new Map<string, Provider>();
  const registry = {
    version: 1,
    register(provider: Provider) {
      const key = `${provider.sessionId}\0${provider.name}`;
      providers.set(key, provider);
      return () => { if (providers.get(key) === provider) providers.delete(key); };
    },
  };
  const symbol = Symbol.for(LIVENESS_KEY);
  (globalThis as any)[symbol] = registry;
  // Pi redirects extension console/stdout writes to stderr. This is an
  // intentional host RPC frame, so write directly to the protocol fd.
  const write = options.write ?? ((value) => writeSync(1, `${JSON.stringify(value)}\n`));
  const owned = (ctx: ExtensionContext) => [...providers.values()].filter(p => p.sessionId === ctx.sessionManager.getSessionId());
  const snapshot = (ctx: ExtensionContext) => {
    const current = owned(ctx);
    return {
      type: 'gms_workflow_state',
      ready: current.some(p => p.name === 'pi-subagents'),
      active: current.some(p => p.isActive()) || !ctx.isIdle() || (ctx.hasPendingMessages?.() ?? false),
    };
  };
  function rpc(method: string, params: unknown, timeoutMs: number): Promise<any> {
    return new Promise((resolve, reject) => {
      const requestId = randomUUID();
      let dispose: (() => void) | undefined;
      const timer = setTimeout(() => { dispose?.(); reject(new Error('Subagent control timed out')); }, timeoutMs);
      dispose = pi.events.on(`subagents:rpc:v1:reply:${requestId}`, (reply: any) => {
        clearTimeout(timer); dispose?.();
        if (reply?.success) resolve(reply.data);
        else reject(new Error('Subagent control request failed'));
      });
      pi.events.emit('subagents:rpc:v1:request', { version: 1, requestId, method, params });
    });
  }
  pi.registerCommand('gms-workflow-status', {
    description: 'Report workflow settlement to the Gmail host',
    handler: async (_args, ctx) => {
      try { write(snapshot(ctx)); }
      catch { write({ type:'gms_workflow_state', active:true, ready:false, error:'Workflow state unavailable' }); }
    },
  });
  pi.registerCommand('gms-workflow-stop', {
    description: 'Stop this session’s outstanding child work',
    handler: async (_args, ctx) => {
      const deadline = Date.now() + (options.stopTimeoutMs ?? 5000);
      const requested = new Set<string>();
      let error: string | undefined;
      try {
        if (!snapshot(ctx).ready) throw new Error('Workflow provider not ready');
        while (owned(ctx).some(p => p.isActive()) && Date.now() < deadline) {
          // The extension resolves this status against its current Pi session;
          // stop independently checks the persisted run's session identity.
          const status = await rpc('status', {}, Math.max(1, deadline - Date.now()));
          const runs = status?.asyncSnapshot?.runs ?? [];
          for (const run of runs) {
            if (!['queued','running'].includes(run.state) || requested.has(run.id)) continue;
            requested.add(run.id);
            try { await rpc('stop', { id:run.id }, Math.max(1, deadline - Date.now())); }
            catch { requested.delete(run.id); }
          }
          if (owned(ctx).some(p => p.isActive())) await new Promise(resolve => setTimeout(resolve, Math.min(50, Math.max(0,deadline-Date.now()))));
        }
        if (owned(ctx).some(p => p.isActive())) error = 'Child cleanup did not complete within the timeout';
      } catch { error = 'Child cleanup could not be confirmed'; }
      try { write({ ...snapshot(ctx), ...(error ? {error} : {}) }); }
      catch { write({type:'gms_workflow_state',active:true,ready:false,error:'Workflow cleanup state unavailable'}); }
    },
  });
  pi.on('session_shutdown', () => {
    providers.clear();
    if ((globalThis as any)[symbol] === registry) delete (globalThis as any)[symbol];
  });
  return { snapshot };
}
