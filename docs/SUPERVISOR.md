# Supervisor

The supervisor is a watchdog that keeps the three long-lived daemons
(`watch`, `update --loop`, `summarize --loop`) alive. Liveness is read
from `job_progress.updated_at` — no pid files on disk.

## Run it once (foreground)

```
gmail-search supervise
```

A daemon is considered alive iff its `job_progress` row has
`status='running'` and `updated_at` within 90s. Stale rows (or missing
rows) trigger a respawn after `--restart-delay` seconds (default 15).

## Run it under systemd-user (auto-start at login)

Drop this at `~/.config/systemd/user/gmail-search-supervisor.service`:

```ini
[Unit]
Description=gmail-search supervisor (keeps watch/update/summarize alive)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/development/gmail-search
ExecStart=%h/anaconda3/bin/gmail-search supervise
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Then:

```
systemctl --user daemon-reload
systemctl --user enable --now gmail-search-supervisor.service
journalctl --user -u gmail-search-supervisor.service -f
```

If you're not using systemd-user, a `@reboot` cron line works too:

```
@reboot /home/ssilver/anaconda3/bin/gmail-search supervise --data-dir /home/ssilver/development/gmail-search/data >> /home/ssilver/development/gmail-search/data/supervisor.log 2>&1
```
