export const REF_PREFIX = "ref://";
export const ATT_PREFIX = "att://";
export const ART_PREFIX = "art://";

// Matches either:
//   [ref:ID]         — bracketed thread citation. ID is normally the full
//                      16-char thread_id; we accept any hex/dash token here
//                      for resilience to model truncation (which we then
//                      try to resolve as a prefix of a known thread_id).
//   [att:123]        — bracketed attachment citation (numeric attachment_id)
//   [art:123]        — bracketed analyst-artifact citation (numeric id from
//                      /api/artifact/<id> — plot PNG, CSV, etc.)
//   <bare hex 8-20>  — bare hex token that's only linkified if it resolves
//                      against a known thread id (to avoid false positives)
const BRACKET_OR_BARE =
  /\[\s*ref:\s*([a-zA-Z0-9_-]+)\s*\]|\[\s*att:\s*(\d+)\s*\]|\[\s*art:\s*(\d+)\s*\]|\b([a-f0-9]{8,20})\b/g;

const resolveAgainstKnown = (id: string, known: readonly string[]): string | null => {
  if (known.includes(id)) return id;
  const matches = known.filter((k) => k.startsWith(id));
  return matches.length === 1 ? matches[0] : null;
};

/**
 * Replace citation markers with markdown links that CitableMarkdown can
 * turn into clickable chips:
 *   [ref:ID]   → `[ID](ref://ID)`   — thread citation, resolves short
 *                  prefixes against the `knownIds` set because Gemini
 *                  sometimes truncates long hex strings.
 *   [att:123]  → `[123](att://123)` — attachment citation; numeric, no
 *                  resolution needed.
 *   bare hex   → linkified only when it exactly (or prefix-)matches a
 *                  known thread id, to avoid turning unrelated hex
 *                  tokens into citations.
 */
export const linkifyRefs = (text: string, knownIds: Iterable<string>): string => {
  const known = Array.from(new Set(knownIds));
  return text.replace(BRACKET_OR_BARE, (match, refId, attId, artId, bareId) => {
    if (attId) {
      return `[${attId}](${ATT_PREFIX}${attId})`;
    }
    if (artId) {
      return `[${artId}](${ART_PREFIX}${artId})`;
    }
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
