import type { NextRequest } from "next/server";
import { CONVERSATION_PATH_RE } from "./conversationUrl";
import { authenticatedUserId, privateHeaders } from "./requestSecurity";

export const publicResponseHeaders = { ...privateHeaders, "Content-Security-Policy": "frame-ancestors 'none'", "X-Frame-Options": "DENY" };
export const isPublicMode = () => process.env.GMS_PUBLIC_ORIGIN !== undefined;
const fail = (status: number, error: string) => Response.json({ error }, { status, headers: publicResponseHeaders });
const anonymous = new Set(["/api/auth/me", "/api/auth/login", "/api/auth/callback", "/api/auth/gmail-callback"]);
const uiPages = new Set(["/", "/search", "/settings"]);
// UI documents: the fixed pages plus one conversation per `/c/<id>`.
const isUiPage = (path: string) => uiPages.has(path) || CONVERSATION_PATH_RE.test(path);
const routes: Array<[RegExp, string[]]> = [
  [/^\/api\/auth\/(me|login|callback|gmail-status|connect-gmail)$/, ["GET"]],
  [/^\/api\/auth\/logout$/, ["POST"]],
  [/^\/api\/chat$/, ["POST"]],
  [/^\/api\/conversations$/, ["GET"]],
  [/^\/api\/conversations\/[a-zA-Z0-9_-]{6,64}$/, ["GET", "PUT", "DELETE"]],
  [/^\/api\/(search|status|users\/me\/sync-status)$/, ["GET"]],
    // Battles are available to owners the server reports as capable; the chat
    // route gates that. Without these the battle runs and then 404s on the vote.
    [/^\/api\/battle\/vote$/, ["POST"]],
    [/^\/api\/battle\/stats$/, ["GET"]],
  [/^\/api\/(thread|thread_lookup|attachment|artifact)\/[^/]+$/, ["GET"]],
  [/^\/api\/attachment\/[^/]+\/meta$/, ["GET"]],
];
export function publicRequestDenied(req: Request): Response | null {
  if (!isPublicMode()) return null;
  let origin: URL;
  try {
    origin = new URL(process.env.GMS_PUBLIC_ORIGIN!);
    if (origin.protocol !== "https:" || origin.origin !== process.env.GMS_PUBLIC_ORIGIN) throw new Error();
  } catch { return fail(503, "Invalid public deployment configuration"); }
  if (req.headers.get("host") !== origin.host) return fail(403, "Invalid request host");
  for (const [key] of req.headers) {
    if (key === "x-user-id" || key === "authorization" || key.startsWith("x-middleware-") || key.startsWith("x-nextjs-") || key === "x-invoke-path" || key === "x-invoke-query" || key === "x-invoke-status" || key === "x-matched-path") return fail(403, "Invalid internal header");
  }
  const path = new URL(req.url).pathname;
  const supplied = req.headers.get("origin");
  if ((supplied && supplied !== origin.origin) || (!supplied && !["GET", "HEAD"].includes(req.method))) return fail(403, "Invalid request origin");
  const site = req.headers.get("sec-fetch-site");
  if (site !== null && !["none", "same-origin", "same-site", "cross-site"].includes(site)) return fail(403, "Invalid request site");
  // OAuth redirects retain cross-site metadata when they reach the landing page.
  // Only explicit UI document navigations may bypass the API fetch-site guard.
  const uiNavigation = isUiPage(path) && ["GET", "HEAD"].includes(req.method)
    && req.headers.get("sec-fetch-mode") === "navigate"
    && req.headers.get("sec-fetch-dest") === "document";
  if (!anonymous.has(path) && !uiNavigation && ["cross-site", "same-site"].includes(site ?? "")) return fail(403, "Invalid request site");
  if (path.startsWith("/api/")) {
    const workerControl = process.env.GMS_FULL_WORKER_ROUTES === "1"
      && ((path === "/api/agent/analyze" && ["GET", "DELETE"].includes(req.method))
        || (path === "/api/auth/connect-gmail" && req.method === "POST")
        || (path === "/api/auth/gmail-callback" && req.method === "GET"));
    if (!workerControl && !routes.some(([pattern, methods]) => pattern.test(path) && methods.includes(req.method))) return fail(404, "Not found");
  } else if (!(isUiPage(path) || path.startsWith("/_next/static/") || path === "/favicon.ico" || path === "/pdf.worker.min.mjs")) return fail(404, "Not found");
  if (req.url.length > 8192) return fail(414, "Request URL too long");
  return null;
}

// Count actual bytes, including chunked bodies, before parsing or chat logging.
export async function publicBodyDenied(req: Request): Promise<Response | null> {
  if (!req.body) return null;
  const limit = 256 * 1024;
  const length = req.headers.get("content-length");
  if (length && (!/^\d+$/.test(length) || Number(length) > limit)) { void req.body.cancel().catch(() => {}); return fail(413, "Request body too large"); }
  const reader = req.clone().body!.getReader();
  let size = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const expired = Symbol("expired");
  const timeout = new Promise<typeof expired>(resolve => {
    timer = setTimeout(() => resolve(expired), 15_000);
  });
  const cancel = () => {
    // Cancel both tee branches: cancelling only the clone leaves the original
    // request retaining buffered bytes and the network source alive.
    void reader.cancel().catch(() => {});
    void req.body?.cancel().catch(() => {});
  };
  try {
    while (true) {
      const item = await Promise.race([reader.read(), timeout]);
      if (item === expired) { cancel(); return fail(408, "Request body timed out"); }
      if (item.done) break;
      size += item.value.byteLength;
      if (size > limit) { cancel(); return fail(413, "Request body too large"); }
    }
  } catch { cancel(); return fail(400, "Invalid request body"); }
  finally { clearTimeout(timer); reader.releaseLock(); }
  return null;
}

// Route-level checks also apply if Next middleware is bypassed or a handler is
// invoked directly. Never trust an inbound header as proof of authentication.
export function publicRoute<C>(handler: (req: NextRequest, ctx: C) => Promise<Response>) {
  return async (req: NextRequest, ctx: C): Promise<Response> => {
    if (!isPublicMode()) return handler(req, ctx);
    const denied = publicRequestDenied(req);
    if (denied) return denied;
    if (!anonymous.has(new URL(req.url).pathname) && !await authenticatedUserId(req)) return fail(401, "Authentication required");
    const bodyDenied = await publicBodyDenied(req);
    if (bodyDenied) return bodyDenied;
    try {
      const response = await handler(req, ctx);
      for (const [key, value] of Object.entries(publicResponseHeaders)) {
        if (key === "Content-Security-Policy" && response.headers.has(key)) response.headers.append(key, value);
        else response.headers.set(key, value);
      }
      return response;
    } catch { return fail(502, "Request could not be completed"); }
  };
}
