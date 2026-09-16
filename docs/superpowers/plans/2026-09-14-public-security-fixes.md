# Public security fixes implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development with explicit file ownership. No commits or public ingress activation are authorized by this plan.

**Goal:** Close confirmed disclosure/ownership holes and prepare fail-closed authentication for a future public deployment.

**Architecture:** Disable unsafe arbitrary SQL at all exposed entry points until an immutable tenant-bound database identity exists. Enforce authenticated ownership for logs and conversations. Public deployment uses an app-specific broker identity contract and explicitly configured origin, while private compatibility is retained until broker registration is provisioned.

**Tech Stack:** FastAPI, PostgreSQL/psycopg, Next.js, silver-oauth identity broker.

## Work and verification

- [ ] Root: Add failing HTTP/tool regression tests proving raw SQL requests are rejected before DB execution. Disable HTTP SQL and MCP/agent SQL tools and remove misleading prompt instructions. Keep fixed parameterized search/read tools available. Run targeted SQL/MCP tests.
- [ ] Root: Add foreign-user conversation PUT regression; require an owner-matching UPSERT RETURNING row before any message mutation, retaining the transaction lock. Restrict global progress to admin and filter user status. Verify synthetic tenant tests.
- [ ] Frontend worker: Add server-side authenticated request guard for chat and log reads; bind logs to authenticated owners and reject legacy unowned logs. Add no-store and active-content response protection. Prove anonymous/wrong-owner requests disclose no log bytes and cannot create chargeable runs. Own web files only.
- [ ] Auth worker: Add opt-in public deployment config that refuses startup without multi-tenant authentication, exact HTTPS origins, separate identity handoff secret and explicit owner allowlist. Implement app-bound broker handoff with required claims, browser-bound one-use nonce, strict CSRF/origin checks and Secure host cookies in public mode. Keep legacy local mode working. Own auth/* and auth tests only; expose startup/middleware hooks for root integration.
- [ ] Root: Add deployment templates that bind web to loopback, refuse unsafe public configuration, and leave hostname/tunnel unpublished. Document broker registration prerequisites and hardening still needed.
- [ ] Review combined code, run relevant Python/Next tests and isolated build; fix findings. Deploy verified containment and ownership/log fixes to existing local services with rollback build retained, then verify live anonymous rejection, disabled SQL and normal search readiness. Do not publish external web ingress or claim full Internet readiness without broker and external acceptance tests.
