/**
 * Walks a tool result payload and collects every `cite_ref` and `thread_id`
 * field anywhere in the structure. Used to validate model citations.
 */
export const collectKnownRefs = (value: unknown, into: Set<string> = new Set()): Set<string> => {
  if (Array.isArray(value)) {
    for (const v of value) collectKnownRefs(v, into);
    return into;
  }
  if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if ((k === "cite_ref" || k === "thread_id") && typeof v === "string" && v) {
        into.add(v);
      } else {
        collectKnownRefs(v, into);
      }
    }
  }
  return into;
};

const REF_BRACKETED = /\[\s*ref:\s*([a-zA-Z0-9_-]+)\s*\]/g;
// Must mirror the same range linkifyRefs.ts uses, otherwise bare IDs
// rendered as chips can slip past validation (Codex #4 alignment bug).
const BARE_HEX = /\b([a-f0-9]{8,20})\b/g;

const looksLikeThreadId = (s: string) => /[a-f]/.test(s) && /[0-9]/.test(s);

const resolves = (id: string, known: ReadonlyArray<string>): boolean => {
  if (known.includes(id)) return true;
  return known.some((k) => k.startsWith(id));
};

export type CitationDiagnostic = {
  broken: string[];
  total: number;
};

const extractRefs = (answer: string): string[] => {
  const refs: string[] = [];
  const seen = new Set<string>();
  for (const m of answer.matchAll(REF_BRACKETED)) {
    const id = m[1];
    if (!seen.has(id)) {
      seen.add(id);
      refs.push(id);
    }
  }
  for (const m of answer.matchAll(BARE_HEX)) {
    const id = m[1];
    if (!looksLikeThreadId(id)) continue;
    if (!seen.has(id)) {
      seen.add(id);
      refs.push(id);
    }
  }
  return refs;
};

/**
 * Pull every citation from the answer text and check it against the known
 * cite_ref / thread_id set. Returns the broken IDs (deduped) so the agent
 * loop can surface them in a correction prompt.
 *
 * `resolveUnknown` is an optional async callback that, given a ref not in
 * the known set, tries to resolve it (e.g. via a DB prefix lookup). If it
 * resolves, we add it to the known set and do NOT flag it as broken — the
 * model may have legitimately referenced a thread from earlier in the
 * conversation.
 */
export const findBrokenRefs = async (
  answer: string,
  knownRefs: Iterable<string>,
  resolveUnknown?: (ref: string) => Promise<string | null>,
): Promise<CitationDiagnostic> => {
  const known = new Set(knownRefs);
  const refs = extractRefs(answer);
  const broken = new Set<string>();

  for (const id of refs) {
    if (resolves(id, [...known])) continue;
    if (resolveUnknown) {
      const resolved = await resolveUnknown(id);
      if (resolved) {
        known.add(resolved);
        continue;
      }
    }
    broken.add(id);
  }

  return { broken: [...broken], total: refs.length };
};
