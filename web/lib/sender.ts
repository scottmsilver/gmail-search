import addrs from "email-addresses";

// Canonical display-name extraction for RFC 5322 addresses. Used
// everywhere a sender / recipient is rendered (ResultRow,
// ThreadDrawer, CitationChip) so parsing rules live in exactly one
// place.
//
// Implementation: `email-addresses` is a pure-JS RFC 5322 parser
// that handles quoted display names, escaped chars, CFWS (folding
// whitespace + comments), address groups, and all the other edge
// cases a hand-rolled regex misses. We hide its API behind a tiny
// wrapper so callers keep using `cleanSender(raw)` and don't have
// to deal with parse results or null returns.
const displayOf = (raw: string): string => {
  if (!raw) return "";
  try {
    const parsed = addrs.parseOneAddress(raw);
    if (parsed && parsed.type === "mailbox") {
      const name = (parsed.name || "").trim();
      if (name) return name;
      return (parsed.address || "").trim();
    }
    if (parsed && parsed.type === "group") {
      if (parsed.name) return parsed.name.trim();
      const first = parsed.addresses?.[0];
      if (first) return (first.name || first.address || "").trim();
    }
  } catch {
    // parseOneAddress is defensive but wrap-to-be-safe.
  }
  // RFC-strict parser rejected this (often because of unquoted
  // commas in the display name — Gmail's "Sasha Torres, MA, BCBA
  // <sasha@x.com>" violates 5322 but we still want to show it
  // nicely). Pull the display name out of everything before the
  // last "<…>" block.
  const loose = raw.match(/^(.+?)\s*<[^<>]+>\s*$/);
  if (loose) return loose[1].replace(/"/g, "").trim();
  return raw.replace(/"/g, "").trim();
};

export const cleanSender = (raw: string): string => displayOf(raw);

// Recipient ("to" / "cc") fields frequently hold comma-separated
// address lists. `cleanRecipients("a@x.com, \"B\" <b@y.com>")` →
// `"a@x.com, B"` — each address parsed, display name preferred when
// present, joined back with ", ".
export const cleanRecipients = (raw: string): string => {
  if (!raw) return "";
  try {
    const list = addrs.parseAddressList(raw) ?? [];
    if (list.length > 0) {
      const out: string[] = [];
      for (const entry of list) {
        if (entry.type === "mailbox") {
          const name = (entry.name || "").trim();
          out.push(name || (entry.address || "").trim());
        } else if (entry.type === "group") {
          if (entry.name) out.push(entry.name.trim());
          for (const m of entry.addresses ?? []) {
            out.push((m.name || m.address || "").trim());
          }
        }
      }
      const filtered = out.filter(Boolean);
      if (filtered.length > 0) return filtered.join(", ");
    }
  } catch {
    // fall through
  }
  // Parser returned nothing useful — fall back to a naive split on
  // commas + per-address clean.
  return raw
    .split(/,\s*/)
    .map((s) => displayOf(s))
    .filter(Boolean)
    .join(", ");
};
