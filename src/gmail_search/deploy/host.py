"""Everything the phases do to the machine, behind one object so tests can
point it at a throwaway install tree with fake runners and probes."""
from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import shlex
import socket
import sqlite3
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .config import DeployConfig
from .runner import Runner

HEALTH_SECONDS = 180


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _http_status(url: str, host_header: str) -> int:
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=15)
    try:
        conn.request('GET', parts.path + (f'?{parts.query}' if parts.query else ''),
                     headers={'Host': host_header, 'X-Forwarded-Proto': 'https'})
        return conn.getresponse().status
    finally:
        conn.close()


def _active_runs(registry: Path) -> int:
    db = sqlite3.connect(f'file:{registry}?mode=ro', uri=True)
    try:
        return db.execute("SELECT count(*) FROM runs WHERE status='active'").fetchone()[0]
    finally:
        db.close()


@dataclass
class Host:
    config: DeployConfig
    runner: Runner = field(default_factory=Runner)
    port_open: Callable[[str, int], bool] = _port_open
    http_status: Callable[[str, str], int] = _http_status
    active_runs: Callable[[Path], int] = _active_runs
    sleep: Callable[[float], None] = time.sleep
    health_seconds: int = HEALTH_SECONDS

    # ── local services ───────────────────────────────────────────────
    def restart(self, *units: str) -> None:
        self.runner.run(['systemctl', '--user', 'restart', *units])

    def unit_active(self, unit: str) -> bool:
        out = self.runner.capture(['systemctl', '--user', 'is-active', unit], check=False)
        return out.strip() == 'active'

    def wait_healthy(self) -> None:
        """Controller and web units active and every health port listening."""
        units = (self.config.services['controller'], self.config.services['web'])
        waited = 0
        while True:
            down = [u for u in units if not self.unit_active(u)]
            down += [str(p) for p in self.config.health_ports if not self.port_open(self.config.health_host, p)]
            if not down:
                return
            if waited >= self.health_seconds:
                raise RuntimeError('not healthy after %ds: %s' % (waited, ', '.join(down)))
            self.sleep(5)
            waited += 5

    # ── the worker ───────────────────────────────────────────────────
    def _ssh_options(self, port_flag: str) -> list[str]:
        w = self.config.worker
        return ['-i', str(w.key), '-o', f'UserKnownHostsFile={w.known_hosts}', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', port_flag, str(w.port)]

    def worker_ssh(self, command: str, *, log: Path | None = None, check: bool = True) -> int:
        w = self.config.worker
        return self.runner.run(['ssh', *self._ssh_options('-p'), f'{w.user}@{w.host}', command], log=log, check=check)

    def worker_upload(self, files: list[Path], remote_dir: str) -> None:
        w = self.config.worker
        self.worker_ssh(f'rm -rf {shlex.quote(remote_dir)} && mkdir -m 700 {shlex.quote(remote_dir)}')
        self.runner.run(['scp', '-q', *self._ssh_options('-P'), *map(str, files), f'{w.user}@{w.host}:{remote_dir}/'])
