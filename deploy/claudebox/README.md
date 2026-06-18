# claudebox local deployment

Runs `psyb0t/claudebox:latest-minimal` in API mode as the Claude backend
for gmail-search. The container exposes the FastAPI on `:8765` and reaches
the host MCP server at `http://host.docker.internal:7878/mcp`.

## Layout

- `docker-compose.yml` — service definition.
- `claude-config/.claude.json` — committed template that wires the
  `gmail-tools` MCP server (HTTP transport).
- `claude-config/.credentials.json` — copied at setup time from
  `~/.claude/.credentials.json`. Gitignored.
- `workspaces/` — per-turn working dirs. Each turn creates a subdir here
  before calling `POST /run`. Gitignored.
- `.env` — holds `CLAUDEBOX_API_TOKEN` (and optionally
  `GMAIL_MCP_SERVICE_TOKEN`, see below). Auto-generated. Gitignored.

## Start / stop

```bash
bash deploy/claudebox/start.sh   # runs setup.sh, brings up compose, waits for health
bash deploy/claudebox/stop.sh    # docker compose down
```

`start.sh` polls `GET /health` (unauthenticated) for up to 60 s, then makes
an authenticated `GET /status` check before reporting `Ready.`.

## Verify it works

```bash
source deploy/claudebox/.env
mkdir -p deploy/claudebox/workspaces/smoketest
curl -sS -X POST http://localhost:8765/run \
  -H "Authorization: Bearer $CLAUDEBOX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Say only the word OK and nothing else.","workspace":"smoketest","model":"sonnet"}' \
  | python3 -m json.tool
```

The workspace directory must exist on the host before calling `/run` — the
container validates `workspace` against `/workspaces/<name>`.

## MCP transport auth (service token)

The host MCP server can enforce transport-token auth on `/mcp`
(`GMAIL_MCP_REQUIRE_TRANSPORT_AUTH=1`). When that is on, an
unauthenticated `/mcp` client gets a 401. claudebox is a **trusted,
multi-tenant** container — the server-side orchestrator registers a
per-run session (`register_session(session_id, user_id=...)`) before each
run, and per-tenant scoping comes from that REGISTERED session, not from
claudebox's static MCP config.

So claudebox must NOT use a tenant-bound transport token (that would pin
every run to a single owner — a cross-tenant bug). Instead it uses a
tenantless **service token** (`aud=mcp-service`): it satisfies enforcement
but carries no identity, so scoping stays on the registered session.

Once enforcement is on:

1. Mint a service token (admin-gated, host-local only):

   ```bash
   curl -sS -X POST http://localhost:7878/admin/service-tokens \
     -H "Authorization: Bearer $GMAIL_MCP_ADMIN_TOKEN" \
     -H "Content-Type: application/json" -d '{}'
   # → {"token":"<jwt>","expires_at":<unix-ts>}   (default TTL: 30 days)
   ```

2. Put it in `deploy/claudebox/.env` as `GMAIL_MCP_SERVICE_TOKEN=<jwt>`.
   `docker-compose.yml` passes that var into the **container env**
   (`GMAIL_MCP_SERVICE_TOKEN: ${GMAIL_MCP_SERVICE_TOKEN:-}`). On every
   `start.sh` run, `register_gmail_tools_mcp()` runs an in-container
   `python3` that reads the token from the container's **own environment**
   and writes the `mcpServers.gmail-tools` entry (with an
   `Authorization: Bearer <token>` header) into the container's
   `~/.claude.json`. If the var is unset, it writes the entry with **no**
   header (so nothing breaks before enforcement is turned on). Re-running
   is idempotent — it overwrites the entry to the desired state.

   SECURITY: the token is **never** placed on a command line. We do not use
   `claude mcp add --header` / `mcp add-json` because those put the bearer
   in the process arg list (visible via `ps` inside the container and
   transiently host-side). The only thing on the `docker exec` argv is the
   `/mcp` URL; the secret is read from the container env by the in-container
   python and written straight to `~/.claude.json` (mode 0600).

Service tokens are long-lived static credentials; rotate by re-minting,
updating `.env`, and re-running `start.sh`. The `/admin/*` endpoints are
loopback-only by default (override with `GMAIL_MCP_ADMIN_ALLOW_REMOTE=1`),
so mint from the host. The server runs uvicorn with `proxy_headers=False`,
so a VM cannot spoof a loopback client address via `X-Forwarded-For` to
reach `/admin/*`.
