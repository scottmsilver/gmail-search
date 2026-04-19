// Date formatting helpers shared across the UI.
//
// - `formatSmartDate`: "2h ago" for anything in the last week, otherwise
//   a calendar date with the year dropped when it matches today's year
//   ("Apr 5" vs "Apr 5, 2024").
// - `formatCalendarDate`: always absolute, same year-smart rule. Used
//   for ranges where "3h ago" would be misleading (e.g. the oldest
//   message in the corpus).
//
// Both rely on Intl.DateTimeFormat via Date.toLocaleDateString so the
// month name follows the user's locale.

const parse = (iso: string): Date | null => {
  const dt = new Date(iso);
  return Number.isNaN(dt.getTime()) ? null : dt;
};

export const formatCalendarDate = (iso: string): string => {
  const dt = parse(iso);
  if (!dt) return "?";
  const sameYear = dt.getFullYear() === new Date().getFullYear();
  return dt.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(sameYear ? {} : { year: "numeric" }),
  });
};

export const formatSmartDate = (iso: string): string => {
  const dt = parse(iso);
  if (!dt) return "?";
  const diffSec = Math.floor((Date.now() - dt.getTime()) / 1000);
  if (diffSec < 60) return `${Math.max(0, diffSec)}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  if (diffSec < 86400 * 7) return `${Math.floor(diffSec / 86400)}d ago`;
  return formatCalendarDate(iso);
};
