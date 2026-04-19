"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS: Array<{ href: string; label: string }> = [
  { href: "/", label: "Chat" },
  { href: "/search", label: "Search" },
];

const isActive = (pathname: string, href: string): boolean => {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
};

const GearIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

export const TopNav = () => {
  const pathname = usePathname() || "/";
  const settingsActive = isActive(pathname, "/settings");
  return (
    <nav
      className="relative flex items-center justify-center gap-1 px-4 h-10 border-b shrink-0"
      style={{ borderColor: "var(--border-subtle)", background: "var(--bg-primary)" }}
    >
      {TABS.map((t) => {
        const active = isActive(pathname, t.href);
        return (
          <Link
            key={t.href}
            href={t.href}
            className={
              active
                ? "px-3 py-1 text-xs font-medium rounded-full"
                : "px-3 py-1 text-xs rounded-full theme-hover"
            }
            style={
              active
                ? { background: "var(--bg-secondary)", color: "var(--fg-primary)" }
                : { color: "var(--fg-secondary)" }
            }
          >
            {t.label}
          </Link>
        );
      })}
      <Link
        href="/settings"
        aria-label="Settings"
        className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-full theme-hover"
        style={{
          color: settingsActive ? "var(--fg-primary)" : "var(--fg-tertiary)",
          background: settingsActive ? "var(--bg-secondary)" : "transparent",
        }}
      >
        <GearIcon />
      </Link>
    </nav>
  );
};
