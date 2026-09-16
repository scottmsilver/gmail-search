// Public deployments compare against configuration, never forwarded Host.
// Private deployments retain non-browser clients but reject cross-origin browsers.
export function rejectUnsafeOrigin(req: Request): Response | null {
  if (["GET", "HEAD", "OPTIONS"].includes(req.method)) return null;
  const configured = process.env.GMS_PUBLIC_ORIGIN;
  const publicMode = configured !== undefined;
  const origin = req.headers.get("origin");
  const url = new URL(req.url);
  const forwardedScheme = req.headers.get("x-forwarded-proto");
  const scheme = !publicMode && ["http", "https"].includes(forwardedScheme ?? "")
    ? `${forwardedScheme}:` : url.protocol;
  const localOrigin = `${scheme}//${req.headers.get("host") ?? url.host}`;
  if ((!origin && !publicMode && !["cross-site", "same-site"].includes(req.headers.get("sec-fetch-site") ?? "")) || (Boolean(origin) && origin === (configured ?? localOrigin))) return null;
  return Response.json({ error: "Invalid request origin" }, {
    status: 403, headers: { "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer" },
  });
}

// Use only after the browser origin has passed the route/middleware guard.
// Internal chat saves and analysis requests must satisfy the backend guard too.
export function backendOriginHeaders(): Record<string, string> {
  return process.env.GMS_PUBLIC_ORIGIN ? { origin: process.env.GMS_PUBLIC_ORIGIN } : {};
}
