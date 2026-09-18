import { pythonApiUrl } from "./config";
import { parseDeepModels, type DeepModels } from "./deepModels";

export const privateHeaders = {
  "Cache-Control": "private, no-store",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
};

// Only a verified backend session can establish ownership. Legacy auth-off
// responses and unavailable/malformed identity responses fail closed.
export type AuthenticatedCaller = {
  id: string;
  // Server-reported: may this caller drive the unrestricted runtimes (model
  // choice, battles, backend selection)? Always true on the private app; on the
  // public origin it follows GMS_FULL_RUNTIME_EMAILS. Never inferred from the
  // deployment, and never taken from the request body.
  fullRuntime: boolean;
  // Server-reported models the isolated-worker service runs; see deepModels.ts.
  deepModels?: DeepModels;
};

export async function authenticatedCaller(req: Request): Promise<AuthenticatedCaller | null> {
  try {
    const response = await fetch(`${pythonApiUrl()}/api/auth/me`, {
      cache: "no-store", redirect: "error", signal: AbortSignal.timeout(5000),
      headers: { cookie: req.headers.get("cookie") ?? "" },
    });
    if (!response.ok) return null;
    const body = await response.json();
    if (body.multi_tenant !== true || typeof body.user?.id !== "string" || !body.user.id.trim()) return null;
    return { id: body.user.id, fullRuntime: body.capabilities?.full_runtime === true,
      deepModels: parseDeepModels(body.capabilities?.deep_models) };
  } catch { return null; }
}

export async function authenticatedUserId(req: Request): Promise<string | null> {
  return (await authenticatedCaller(req))?.id ?? null;
}

export const unauthenticated = () => Response.json(
  { error: "Authentication required" }, { status: 401, headers: privateHeaders },
);

export function downloadHeaders(upstream: Headers): Headers {
  const headers = new Headers(privateHeaders);
  for (const name of ["content-type", "content-length", "content-disposition", "content-security-policy", "referrer-policy", "cross-origin-resource-policy"]) {
    const value = upstream.get(name);
    if (value) headers.set(name, value);
  }
  // Only passive raster images, PDFs and plain text may render inline.
  const mime = (headers.get("content-type") ?? "").split(";")[0].trim().toLowerCase();
  if (!["image/png", "image/jpeg", "image/gif", "image/webp", "image/avif", "application/pdf", "text/plain"].includes(mime)) {
    const disposition = headers.get("content-disposition") ?? "";
    headers.set("content-disposition", `attachment${disposition.includes(";") ? disposition.slice(disposition.indexOf(";")) : ""}`);
  }
  const csp = headers.get("content-security-policy");
  headers.set("content-security-policy", [csp, "sandbox; default-src 'none'"].filter(Boolean).join(", "));
  headers.set("cross-origin-resource-policy", "same-origin");
  return headers;
}
