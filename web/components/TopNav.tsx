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

export const TopNav = () => {
  const pathname = usePathname() || "/";
  return (
    <nav
      className="flex items-center justify-center gap-1 px-4 h-10 border-b shrink-0"
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
    </nav>
  );
};
