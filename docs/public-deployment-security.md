# Public Gmail Search deployment

Status: **deployed, owner-only** at https://gms.oursilverfamily.com on 2026-09-14.
Real owner Google sign-in from an off-LAN browser is the remaining manual acceptance check.
Automated signed-handoff tests are not a substitute for that check or an independent penetration test.

## Access boundary

- Only `scottmsilver@gmail.com` is admitted. Private invitations do not grant public access.
- The broker has a separate `gmail-search` registration, exact callback and independent secret. The existing terminal registration was preserved and tested.
- Handoffs require the configured issuer, audience, verified email, browser-bound one-use nonce and token ID, and at most60-second validity.
- Sessions are opaque, server-side, one-hour, Secure/HttpOnly/host-only cookies. Logout revokes them. Restart invalidates sessions; one actual API worker is enforced.
- Public API routes and methods are explicitly allowed. Admin, jobs, debug logs/workspaces, SQL and direct agent endpoints are not exposed through the web boundary. All mail reads and conversation changes require ownership.
- Browser Host/Origin and internal-header checks apply in middleware and route handlers. Response caching is disabled; Cloudflare probes reported DYNAMIC. Frames are denied and referrers suppressed.
- Actual request bytes and read time are bounded. Public backend admission separates anonymous capacity from authenticated traffic, caps requests and concurrent chats, and bounds SQL statement time.

## Public chat capabilities

Public chat uses a bounded Gemini retrieval loop: search, structured filters, thread reading, facts and attachment text/metadata. Model-supplied identity overrides and undeclared tools are rejected. The public process never launches Pi/Claude containers, creates shared workspaces, runs shell/code, accesses arbitrary URLs or publishes files. Runtime selection supplied by the browser is ignored publicly. Model APIs still process the mail selected by authorized retrieval, as in the existing product.

This closes the public shared-agent isolation problem by removing arbitrary execution from that deployment. The private LAN app retains its existing agent features; those shared shell-enabled runtimes are **not** cleared for public multi-user exposure.

Model Markdown cannot load images automatically. Email HTML blocks remote resources. Active downloads receive attachment disposition, nosniff and sandbox CSP. External links require deliberate user navigation and suppress referrers.

## Database isolation

Public API uses dedicated `gmail_search_public`, with no superuser, role-creation or RLS-bypass privileges and no privileged role memberships. Fixed owner-pinned restrictive policies prevent changing `app.user_id` from widening access. The role reads only allowed mail tables and writes only conversation/session/event/cost records. It cannot modify mail, access the global query cache or regain the application superuser role. Public login requires an already connected mailbox and never creates users.

The initial broad extension ACL proposal was rejected by automatic approval review. A narrower, approved migration changed only the named extension objects in `deploy/public/restrict-extension-acls.sql`, preserving access for the audited existing roles. Read-only spatial/index metadata is explicitly allowed. See [database operations](../deploy/public/DATABASE.md).

## Running services and releases

- `gmail-search-public-api.service`: loopback8091, frozen Python source, restricted DB login, one worker.
- `gmail-search-public-web.service`: loopback3001, verified Next.js build.
- `cloudflared-gmail-search-web.service`: dedicated tunnel `gmail-search-web`, ID `cb9237c8-2a7e-4ebc-be6c-50404f430862`, exact hostname and unmatched404.
- No inbound router/firewall port was opened. Existing LAN3000/8090, public MCP and terminal services were preserved.
- Release: `~/.local/share/gmail-search/public-releases/security-20260914-final`; active symlink `public-current`.
- Secrets: mode0600 files under `~/.config/gmail-search/`, outside releases and source control. Public web receives no database, broker handoff or model-provider secrets.
- Service templates and ingress config: `deploy/public/`. All three services are enabled for restart/login persistence.
- Numerical search libraries require over600 threads in this deployment; TasksMax1024, bounded library pools, CPUQuota400%, MemoryMax16G, eight request worker threads. The lower initial task cap was tested, failed startup and was corrected before ingress activation.

## Verification performed

- Broker tests:8 passed; both live application start URLs redirect to Google and reject unregistered callbacks.
- Final public backend/database/runtime/auth suite:82 passed. Broader auth/containment compatibility suite:123 passed before final deployment refinements.
- Frontend:3 public-boundary,1 Markdown,19 chat-route and5 battle tests passed; TypeScript and production build passed.
- Independent code review covered broker/runtime/route boundaries; reported rate-limit and slow-body issues were fixed.
- Live restricted DB login: owner rows visible, other-user rows hidden after tenant-setting manipulation, privilege escalation denied, global query cache denied.
- Live local and public HTTPS: anonymous mail401, forbidden logs/admin/SQL404, foreign Origin403, forged internal-header403, middleware-bypass attempt403, private/no-store responses.
- Synthetic signed owner handoff307, replay401, valid owner session/search/conversations200, browser chat200 with real Gemini synthetic completion, logout200 and revoked session401.
- Synthetic runtime request specified an alternate backend; the server still selected the public retrieval runtime.
- No private mail bodies or credentials were printed by deployment checks. Synthetic handoff and no-mail chat probes were used; model-backed search probes requested only a synthetic canary query.

## Remaining manual check

Owner: open https://gms.oursilverfamily.com on a phone with Wi-Fi off, sign in through Google, and confirm search/chat and a source citation open correctly. Report any error. Do not add public users or enable shell/code tools without redesigning and testing their isolation boundary.

## Emergency rollback

First stop public ingress:

```sh
systemctl --user disable --now cloudflared-gmail-search-web.service
```

Then stop or repair the public services. Keep the LAN app available. Do not switch public ingress to the private app or restore a release with anonymous log access. Broker rollback must preserve terminal registration; leave the dedicated secret private. Retain restricted DB policies rather than restoring unsafe public grants.
