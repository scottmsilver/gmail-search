export const REF_PREFIX = "ref://";

const BRACKET_OR_BARE = /\[\s*ref:\s*([a-zA-Z0-9_-]+)\s*\]|\b([a-f0-9]{14,20})\b/g;

/**
 * Replace [ref:ID] markers AND bare known thread IDs with markdown links to
 * `ref://ID`. The model is supposed to use the bracket form, but it sometimes
 * emits the raw 16-char hex ID inline; we still want those to render as chips.
 *
 * Bare IDs are only linkified when they appear in `knownIds` to avoid turning
 * unrelated hex strings into citations.
 */
export const linkifyRefs = (text: string, knownIds: Iterable<string>): string => {
  const known = new Set(knownIds);
  return text.replace(BRACKET_OR_BARE, (match, refId, bareId) => {
    const id = refId ?? bareId;
    if (refId) {
      // Bracketed form — always link, even if not in knownIds (chip will fetch).
      return `[${id}](${REF_PREFIX}${id})`;
    }
    if (known.has(id)) {
      return `[${id}](${REF_PREFIX}${id})`;
    }
    return match;
  });
};
