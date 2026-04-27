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
- `.env` — holds `CLAUDEBOX_API_TOKEN`. Auto-generated. Gitignored.

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
