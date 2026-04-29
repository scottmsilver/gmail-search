"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { AvatarMenu } from "@/components/AvatarMenu";
import { cn } from "@/lib/utils";

const TABS: Array<{ href: string; label: string }> = [
  { href: "/", label: "Chat" },
  { href: "/search", label: "Search" },
  { href: "/inbox", label: "Inbox" },
  { href: "/priority", label: "Priority" },
];

const isActive = (pathname: string, href: string): boolean => {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
};

export const TopNav = () => {
  const pathname = usePathname() || "/";
  return (
    // z-50 on the nav itself so the avatar dropdown's stacking context
    // (created by the wrapper's translateY transform below) sits above
    // chat content in <main>. Without this the menu renders correctly
    // but is painted UNDER chat bubbles since <main> appears later in
    // source order with its own auto-stacking.
    <nav className="relative z-50 flex h-10 shrink-0 items-center justify-center gap-1 border-b bg-background px-4">
      {TABS.map((t) => {
        const active = isActive(pathname, t.href);
        return (
          <Link
            key={t.href}
            href={t.href}
            className={cn(
              "rounded-full px-3 py-1 text-xs transition-colors",
              active
                ? "bg-secondary font-medium text-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
            )}
          >
            {t.label}
          </Link>
        );
      })}
      {/* Avatar (or gear in single-pool mode) sits at the far right;
          its dropdown holds Settings + Sign out so the right side is
          a single affordance. */}
      <div className="absolute right-2 top-1/2 -translate-y-1/2">
        <AvatarMenu />
      </div>
    </nav>
  );
};
