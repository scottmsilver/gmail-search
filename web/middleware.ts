import { NextRequest, NextResponse } from "next/server";
import { isPublicMode, publicRequestDenied, publicResponseHeaders } from "./lib/publicBoundary";
import { rejectUnsafeOrigin } from "./lib/originSecurity";

export function middleware(req: NextRequest) {
  if (!isPublicMode() && !new URL(req.url).pathname.startsWith("/api/")) return NextResponse.next();
  const denied = publicRequestDenied(req) ?? rejectUnsafeOrigin(req);
  if (denied) return denied;
  const response = NextResponse.next();
  if (isPublicMode()) for (const [key, value] of Object.entries(publicResponseHeaders)) response.headers.set(key, value);
  response.headers.set("Cache-Control", "private, no-store");
  response.headers.set("X-Content-Type-Options", "nosniff");
  response.headers.set("Referrer-Policy", "no-referrer");
  return response;
}

export const config = { matcher: "/:path*" };
