import { pythonApiUrl } from "./config";

export const privateHeaders = {
  "Cache-Control": "private, no-store",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
};

// Only a verified backend session can establish ownership. Legacy auth-off
// responses and unavailable/malformed identity responses fail closed.
export async function authenticatedUserId(req: Request): Promise<string | null> {
  try {
    const response = await fetch(`${pythonApiUrl()}/api/auth/me`, {
      cache: "no-store", redirect: "error", signal: AbortSignal.timeout(5000),
      headers: { cookie: req.headers.get("cookie") ?? "" },
    });
    if (!response.ok) return null;
    const body = await response.json();
    return body.multi_tenant === true && typeof body.user?.id === "string" && body.user.id.trim()
      ? body.user.id : null;
  } catch { return null; }
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
