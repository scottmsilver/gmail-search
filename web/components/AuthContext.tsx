// React context that carries "who am I and is the auth wall on?" so
// the AvatarMenu can render without re-fetching what AuthGate already
// fetched.
//
// Three values matter to consumers:
//   * `multiTenant` — false means single-pool legacy mode; the avatar
//     menu hides itself.
//   * `user` — null when multi-tenant is on but the auth probe found
//     no session; AuthGate is showing the sign-in screen so consumers
//     of this context shouldn't render at all.
//   * `signOut` — POST /api/auth/logout, then full reload so AuthGate
//     re-runs against the now-empty session.

"use client";

import { createContext, useContext } from "react";

export type AuthUser = {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
};

export type AuthContextValue = {
  multiTenant: boolean;
  user: AuthUser | null;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export const AuthProvider = AuthContext.Provider;

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    // Reachable only if a consumer renders outside <AuthGate>. Surface
    // it noisily so we don't silently render an avatar menu against
    // stale/undefined identity.
    throw new Error("useAuth() called outside of <AuthGate>");
  }
  return ctx;
}
