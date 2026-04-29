// Top-right account menu. Works in both modes:
//   * multi-tenant on, signed-in: avatar trigger → dropdown with
//     name + email + Settings + Sign out.
//   * multi-tenant off (single-pool legacy): gear trigger → dropdown
//     with just Settings. Same affordance shape so the UI doesn't
//     have a different right-side element across modes.

"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { useAuth } from "@/components/AuthContext";
import { cn } from "@/lib/utils";

const GearIcon = () => (
  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

export function AvatarMenu() {
  const { multiTenant, user, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Click-outside + Escape close. Native <details> would do this for
  // free, but Tailwind's `list-none` breaks <summary> click handling
  // in some browsers, so we drive it with state instead.
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      const el = wrapperRef.current;
      if (!el) return;
      if (e.target instanceof Node && !el.contains(e.target)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const isSignedIn = multiTenant && !!user;
  const fallbackInitial = (user?.name?.[0] ?? user?.email[0] ?? "?").toUpperCase();

  return (
    <div ref={wrapperRef} className="relative">
      <button
        type="button"
        aria-label={isSignedIn ? "Account menu" : "App menu"}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "flex h-7 w-7 cursor-pointer items-center justify-center",
          "overflow-hidden rounded-full border border-border bg-secondary text-xs",
          "font-semibold text-secondary-foreground transition hover:opacity-90",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        )}
      >
        {isSignedIn ? (
          user!.picture ? (
            // Plain <img> — these come from Google's CDN with paths
            // next.config doesn't allowlist for next/image.
            // eslint-disable-next-line @next/next/no-img-element
            <img src={user!.picture} alt="" className="h-full w-full object-cover" />
          ) : (
            fallbackInitial
          )
        ) : (
          <GearIcon />
        )}
      </button>
      {open ? (
        <div
          role="menu"
          className={cn(
            "absolute right-0 top-full z-50 mt-2 w-64 rounded-md border border-border bg-popover",
            "p-3 text-sm shadow-md text-popover-foreground",
          )}
        >
          {isSignedIn ? (
            <div className="flex items-center gap-3 pb-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-full border border-border bg-secondary text-sm font-semibold">
                {user!.picture ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={user!.picture} alt="" className="h-full w-full object-cover" />
                ) : (
                  fallbackInitial
                )}
              </div>
              <div className="min-w-0 flex-1">
                {user!.name ? <div className="truncate font-medium">{user!.name}</div> : null}
                <div className="truncate text-xs text-muted-foreground">{user!.email}</div>
              </div>
            </div>
          ) : null}
          <div className={cn("space-y-1", isSignedIn ? "border-t border-border pt-2" : "")}>
            <Link
              href="/settings"
              onClick={() => setOpen(false)}
              className={cn(
                "block rounded px-2 py-1.5 text-xs",
                "transition hover:bg-accent hover:text-accent-foreground",
              )}
            >
              Settings
            </Link>
            {isSignedIn ? (
              <button
                type="button"
                onClick={() => {
                  setOpen(false);
                  void signOut();
                }}
                className={cn(
                  "w-full rounded px-2 py-1.5 text-left text-xs",
                  "transition hover:bg-accent hover:text-accent-foreground",
                )}
              >
                Sign out (and switch account)
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
