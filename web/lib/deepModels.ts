import type { DeepBackend } from "./config";

// Server-reported (backend -> models) this deployment actually runs, first model
// the default. The isolated-worker service reports it from /api/auth/me; when it
// is present the picker offers exactly these and nothing is substituted.
export type DeepModels = Partial<Record<DeepBackend, string[]>>;

const BACKENDS: readonly DeepBackend[] = ["pi", "claude_code"];

export function parseDeepModels(value: unknown): DeepModels | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const served: DeepModels = {};
  for (const [backend, models] of Object.entries(value)) {
    if (!(BACKENDS as readonly string[]).includes(backend)) continue;
    if (!Array.isArray(models) || !models.length || models.some(m => typeof m !== "string" || !m)) return undefined;
    served[backend as DeepBackend] = models as string[];
  }
  return Object.keys(served).length ? served : undefined;
}

// The served choice nearest the request: its backend if served (else Pi, else
// the first), and its model if served for that backend (else the default).
export function servedChoice(served: DeepModels, backend?: string, model?: string): { backend: DeepBackend; model: string } {
  const pick = BACKENDS.find(b => b === backend && served[b])
    ?? (served.pi ? "pi" : BACKENDS.find(b => served[b]))!;
  const models = served[pick]!;
  return { backend: pick, model: model && models.includes(model) ? model : models[0] };
}
