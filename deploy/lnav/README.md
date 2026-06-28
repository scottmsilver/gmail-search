# Browsing the logs with lnav

[lnav](https://lnav.org) is a terminal log browser: it merges every `data/*.log`
into one timestamp-ordered view, live-tails, filters, and runs SQL over the logs.
It's the right-sized tool for a single host (no services, no ports).

## Install

```bash
# Debian/Ubuntu
sudo apt install lnav
# macOS
brew install lnav
# or a standalone musl binary (no root): https://github.com/tstack/lnav/releases
```

## Use it

```bash
# Install this format so JSON logs become queryable columns:
mkdir -p ~/.lnav/formats/installed
cp deploy/lnav/gmail-search.json ~/.lnav/formats/installed/

# Merge + live-tail every log:
lnav data/*.log
```

For the richest experience (column extraction + SQL), run the services with
**`GMS_LOG_JSON=1`** so the lines are JSON — this format definition then parses
`trace_id`, `status`, `dur_ms`, `tool`, `session_id`, `path`, etc. into fields.
(Plain text logs still merge/tail/filter fine, just without typed columns.)

### Handy lnav commands (press `:` for commands, `;` for SQL)

```
:filter-in trace_id=0d5cd23e9cf64fc38cc0bbab934f0efe   # one request, all hops
:filter-in event=mcp_tool_call                          # only MCP tool calls
;SELECT path, status, dur_ms FROM gmail_search_json
   WHERE dur_ms > 1000 ORDER BY dur_ms DESC;            # slowest requests
;SELECT tool, count(*) FROM gmail_search_json
   WHERE event='mcp_tool_call' GROUP BY tool;           # tool call counts
```

## Want a real web UI instead?

If you later want a browser (charts, remote access, retention), the lightest
self-hosted option is [OpenObserve](https://openobserve.ai) — a single Go binary,
local-disk storage, logs+metrics+traces in one process, ingests JSON over HTTP.
Overkill for one user today; lnav covers it.
