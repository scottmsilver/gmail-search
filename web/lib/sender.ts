import * as addrs from "email-addresses";

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
export const cleanSender = (raw: string): string => {
  if (!raw) return "";
  try {
    // `parseOneAddress` returns:
    //   { type: "mailbox", name: "Scott Silver", address: "scott@…", … }
    //   { type: "group", …, addresses: [...] }
    //   null — unparseable
    const parsed = addrs.parseOneAddress({ input: raw, partial: true });
    if (parsed && parsed.type === "mailbox") {
      const name = (parsed.name || "").trim();
      if (name) return name;
      return (parsed.address || "").trim();
    }
    if (parsed && parsed.type === "group") {
      // Groups are rare in practice but the spec allows them. Take the
      // display name of the group itself, or fall back to the first
      // member's name/address.
      if (parsed.name) return parsed.name.trim();
      const first = parsed.addresses?.[0];
      if (first) return (first.name || first.address || "").trim();
    }
  } catch {
    // parseOneAddress is defensive but wrap-to-be-safe.
  }
  // Unparseable — return the raw string stripped of the worst offenders.
  return raw.replace(/"/g, "").trim();
};
