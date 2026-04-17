export const REF_PREFIX = "ref://";

const BRACKET_OR_BARE = /\[\s*ref:\s*([a-zA-Z0-9_-]+)\s*\]|\b([a-f0-9]{8,20})\b/g;

const resolveAgainstKnown = (id: string, known: readonly string[]): string | null => {
  if (known.includes(id)) return id;
  const matches = known.filter((k) => k.startsWith(id));
  return matches.length === 1 ? matches[0] : null;
};

/**
 * Replace [ref:ID] markers AND bare known thread IDs with markdown links to
 * `ref://ID`. The model sometimes emits a truncated prefix of the real ID
 * (Gemini Flash Lite is sloppy with long hex strings); we prefix-match against
 * the known set when the exact ID isn't found.
 *
 * Bare IDs are only linkified when they resolve to a known thread, to avoid
 * turning unrelated hex strings into citations.
 */
export const linkifyRefs = (text: string, knownIds: Iterable<string>): string => {
  const known = Array.from(new Set(knownIds));
  return text.replace(BRACKET_OR_BARE, (match, refId, bareId) => {
    const raw = (refId ?? bareId) as string;
    const resolved = resolveAgainstKnown(raw, known);
    if (resolved) {
      return `[${resolved}](${REF_PREFIX}${resolved})`;
    }
    if (refId) {
      // Bracketed ref but unknown — still emit a chip; the chip can fetch lazily.
      return `[${raw}](${REF_PREFIX}${raw})`;
    }
    return match;
  });
};
