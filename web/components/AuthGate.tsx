// Client-side auth gate. Mounts at the top of the app, fetches
// /api/auth/me, and either renders children or shows a sign-in CTA.
//
// Three terminal states from the backend:
//   * 200 + {multi_tenant: false, user: null} — single-pool legacy
//     mode. Render children unconditionally; no auth wall is in
//     effect on the server.
//   * 200 + {multi_tenant: true, user: {…}} — multi-tenant on,
//     signed in. Render children + provide identity context for
//     things like the avatar menu.
//   * 401 — multi-tenant on, NOT signed in. Show the sign-in button.
//
// The sign-in button is a plain <a> (top-level GET nav). SameSite=Lax
// session cookies survive top-level navigation but not XHR/fetch from
// other origins, so a real <a> is the right primitive here.

"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { AuthProvider, type AuthUser } from "@/components/AuthContext";

type AuthState =
  | { kind: "checking" }
  | { kind: "open" } // multi-tenant off — anyone can use the app
  | { kind: "signed-in"; user: AuthUser }
  | { kind: "signed-out" }
  | { kind: "error"; detail: string };

export function AuthGate({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({ kind: "checking" });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/auth/me", { cache: "no-store", credentials: "include" });
        if (cancelled) return;
        if (res.status === 401) {
          setState({ kind: "signed-out" });
          return;
        }
        if (!res.ok) {
          setState({ kind: "error", detail: `auth probe returned ${res.status}` });
          return;
        }
        const body = await res.json();
        if (!body.multi_tenant) {
          setState({ kind: "open" });
        } else if (body.user) {
          const u = body.user;
          setState({
            kind: "signed-in",
            user: {
              id: u.id,
              email: u.email,
              name: u.name ?? null,
              picture: u.picture ?? null,
            },
          });
        } else {
          setState({ kind: "signed-out" });
        }
      } catch (err) {
        if (cancelled) return;
        setState({ kind: "error", detail: String(err) });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Sign-out: clear the cookie, then full reload. We do a hard reload
  // (not router.push) so AuthGate's effect re-runs against the now-
  // empty session and the broker round-trip starts cleanly. Going
  // straight to /api/auth/login also gives us account-switch behavior
  // — Google prompts to choose an account when multiple are signed in.
  const signOut = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
    } catch {
      // Ignore — even if the POST fails, the next /me probe will
      // trip the sign-out screen because the session is already
      // server-stateless. Reload anyway.
    }
    window.location.assign("/");
  }, []);

  // Memoize so consumers don't re-render every time AuthGate does.
  const ctxOpen = useMemo(
    () => ({ multiTenant: false, user: null, signOut }),
    [signOut],
  );
  const ctxSignedIn = useMemo(
    () =>
      state.kind === "signed-in"
        ? { multiTenant: true, user: state.user, signOut }
        : null,
    [state, signOut],
  );

  if (state.kind === "checking") {
    return <FullPageMessage title="Loading…" />;
  }

  if (state.kind === "error") {
    return <FullPageMessage title="Couldn't reach the auth service" body={state.detail} />;
  }

  if (state.kind === "signed-out") {
    const here =
      typeof window !== "undefined"
        ? window.location.pathname + window.location.search
        : "/";
    const loginHref = `/api/auth/login?return_url=${encodeURIComponent(here)}`;
    return (
      <FullPageMessage
        title="Sign in to Gmail Search"
        body="This app is invite-only. Sign in with your Google account to continue."
        action={
          <a
            href={loginHref}
            className="inline-flex items-center gap-2 rounded-md border border-border bg-foreground px-4 py-2 text-sm font-medium text-background hover:opacity-90"
          >
            Sign in with Google
          </a>
        }
      />
    );
  }

  // Open mode (multi-tenant off): no signed-in user, but the app is
  // usable. Provide a no-op context so AvatarMenu can opt out cleanly
  // via the multiTenant flag.
  if (state.kind === "open") {
    return <AuthProvider value={ctxOpen}>{children}</AuthProvider>;
  }

  // signed-in
  return <AuthProvider value={ctxSignedIn!}>{children}</AuthProvider>;
}

function FullPageMessage({
  title,
  body,
  action,
}: {
  title: string;
  body?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-8 text-center">
      <h1 className="text-xl font-semibold">{title}</h1>
      {body ? <p className="max-w-md text-sm text-muted-foreground">{body}</p> : null}
      {action}
    </div>
  );
}
