# Supervisor

The supervisor is a watchdog that keeps the three long-lived daemons
(`watch`, `update --loop`, `summarize --loop`) alive. Liveness is read
from `job_progress.updated_at` — no pid files on disk.

The venv used below is created with `uv sync --extra dev` in the repo root.

## Run it once (foreground)

```
gmail-search supervise
```

A daemon is considered alive iff its `job_progress` row has
`status='running'` and `updated_at` within 90s. Stale rows (or missing
rows) trigger a respawn after `--restart-delay` seconds (default 15).

## Run it under systemd-user (auto-start at login)

Drop this at `~/.config/systemd/user/gmail-search-supervise.service`:

```ini
[Unit]
Description=gmail-search supervisor (keeps watch/update/summarize alive)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/development/gmail-search
ExecStart=%h/development/gmail-search/.venv/bin/gmail-search supervise
Restart=on-failure
RestartSec=10
# Inherited by every spawned daemon. `crawl` drives a Chromium pool through
# Playwright, whose drivers cost pipe descriptors; on systemd's default of
# 1024 a driver leak exhausted the daemon in 19 minutes and wedged crawling
# for four days (2026-09-15). Do not omit this.
LimitNOFILE=65536

[Install]
WantedBy=default.target
```

Then:

```
systemctl --user daemon-reload
systemctl --user enable --now gmail-search-supervise.service
journalctl --user -u gmail-search-supervise.service -f
```

If you're not using systemd-user, a `@reboot` cron line works too:

```
@reboot /home/ssilver/development/gmail-search/.venv/bin/gmail-search supervise --data-dir /home/ssilver/development/gmail-search/data >> /home/ssilver/development/gmail-search/data/supervisor.log 2>&1
```
