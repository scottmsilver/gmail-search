"""The owner daemons as systemd runs them: which units run the owner's code,
how each one answers when healthy, and a Machine that asks systemd, ss and
/proc about them, behind one object so tests fake it. Shared by the one-time
moves (one_checkout, owner_cutover) and the deployer's owner track (#52)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import shlex
import subprocess
import time
from typing import Callable

from . import checkout_checks as checks
from .host import _active_runs, _port_open
from .runner import Runner

say = lambda message: print(f'==> {message}', flush=True)  # noqa: E731

# The units that run the checkout's own code. Started backend first, so mcp
# and web never come up against a serve still on the old code, and supervise
# last: its children (watch, update, reindex, summarize, crawl) write
# constantly and every one of them runs init_db. Stopped in the reverse.
OWNER_UNITS = ('gmail-search-serve.service', 'gmail-search-mcp.service',
               'gmail-search-web.service', 'gmail-search-supervise.service')
WATCHDOG_TIMER = 'gmail-search-serve-watchdog.timer'
WEB_UNIT = 'gmail-search-web.service'
# Share the checkout's .venv or web/node_modules but run a release's code; the
# switch must leave them running (same MainPID before and after).
SHARING_UNITS = ('gmail-search-invited-api.service', 'gmail-search-public-web.service')
# Each unit's own readiness request and the answers that mean healthy (mcp:
# up, with its OAuth gate on); the port is read from the running unit.
PROBES = {'gmail-search-serve.service': ('GET', '/healthz?ready=1', {200}),
          'gmail-search-mcp.service': ('POST', '/mcp', {401}),
          'gmail-search-web.service': ('GET', '/', {200})}
HEALTH_SECONDS = {'gmail-search-serve.service': 300}
DEFAULT_HEALTH_SECONDS = 120


# ── the machine, behind one object so tests fake it ──────────────────
def _listening(pid: int) -> list[tuple[str, int]]:
    out = subprocess.run(['ss', '-ltnpH'], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        if f'pid={pid},' in line:
            addr = line.split()[3]
            host, _, port = addr.rpartition(':')
            found.append((host.strip('[]'), int(port)))
    return found


def _process_tree(pid: int) -> list[int]:
    out = subprocess.run(['ps', '-o', 'pid=', '--ppid', str(pid)], capture_output=True, text=True).stdout
    kids = [int(p) for p in out.split()]
    return [pid] + [q for k in kids for q in _process_tree(k)]


def _http_status(method: str, url: str) -> int:
    import http.client
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=30)
    try:
        conn.request(method, parts.path + (f'?{parts.query}' if parts.query else ''))
        return conn.getresponse().status
    except OSError:
        return 0
    finally:
        conn.close()


@dataclass
class Machine:
    checkout: Path
    runner: Runner = field(default_factory=Runner)
    dry_run: bool = False
    health_host: str = ''
    registry: Path | None = None
    listening: Callable[[int], list] = _listening
    process_tree: Callable[[int], list] = _process_tree
    http_status: Callable[[str, str], int] = _http_status
    port_open: Callable[[str, int], bool] = _port_open
    active_runs: Callable[[Path], int] = _active_runs
    sleep: Callable[[float], None] = time.sleep

    def git(self, *args, check=True) -> str:
        # --no-optional-locks: a read (status) must not rewrite the checkout's index.
        return self.runner.capture(['git', '--no-optional-locks', '-C', self.checkout, *args], check=check).strip()

    def act(self, args, **kw) -> None:
        """A command that changes something: printed always, run unless dry."""
        where = f" (in {kw['cwd']})" if kw.get('cwd') else ''
        say(f"{'would run' if self.dry_run else 'run'}: {shlex.join(map(str, args))}{where}")
        if not self.dry_run:
            self.runner.run(args, **kw)

    def unit(self, unit: str, prop: str) -> str:
        return self.runner.capture(['systemctl', '--user', 'show', '-p', prop, '--value', unit],
                                   check=False).strip()

    def main_pid(self, unit: str) -> int:
        return int(self.unit(unit, 'MainPID') or 0)

    def unit_tree(self, unit: str) -> list[int]:
        pid = self.main_pid(unit)
        return self.process_tree(pid) if pid else []

    def unit_ports(self, unit: str) -> list[tuple[str, int]]:
        return sorted({a for p in self.unit_tree(unit) for a in self.listening(p)})

    def unit_path(self, unit: str) -> str:
        """PATH as the unit runs with it, so a build uses the same node."""
        found = re.search(r'(?:^|\s)PATH=(\S+)', self.unit(unit, 'Environment'))
        return found.group(1) if found else os.environ.get('PATH', '')


# ── health ───────────────────────────────────────────────────────────
def _url(m: Machine, host: str, port: int, path: str) -> str:
    host = m.health_host if host in ('0.0.0.0', '*', '::', '') else host
    return f'http://{host}:{port}{path}'


def probe_unit(m: Machine, unit: str) -> dict:
    """What the unit answers now: its ports and each one's status for the
    unit's own readiness request."""
    method, path, _ = PROBES.get(unit, (None, None, None))
    ports = m.unit_ports(unit)
    return {'pid': m.main_pid(unit), 'ports': ports, 'children': len(m.unit_tree(unit)) - 1,
            'status': {str(port): m.http_status(method, _url(m, host, port, path))
                       for host, port in ports} if method else {}}


def baseline(m: Machine, units=(*OWNER_UNITS, *SHARING_UNITS)) -> dict:
    return {u: probe_unit(m, u) for u in units}


def wait_like_before(m: Machine, unit: str, before: dict) -> str:
    """Poll until the unit is active, on the same ports, and each answers the
    status it answered before the switch (plus children for supervise)."""
    deadline = HEALTH_SECONDS.get(unit, DEFAULT_HEALTH_SECONDS)
    for _ in range(deadline // 5):
        if m.unit(unit, 'ActiveState') == 'active':
            now_ = probe_unit(m, unit)
            same = {str(p): s for p, s in now_['status'].items()} == before['status']
            kids = now_['children'] > 0 if unit == 'gmail-search-supervise.service' else True
            if same and [list(p) for p in now_['ports']] == [list(p) for p in before['ports']] and kids:
                return f"{unit} healthy: pid {now_['pid']}, {now_['status'] or 'no probe'}"
        m.sleep(5)
    raise RuntimeError(f'{unit} not healthy after {deadline}s (before: {before}); '
                       f'journalctl --user -u {unit} -n 50')


def _baseline_ok(unit: str, b: dict) -> bool:
    if not b['pid']:
        return False
    if unit in PROBES:
        return bool(b['status']) and all(s in PROBES[unit][2] for s in b['status'].values())
    return unit != 'gmail-search-supervise.service' or b.get('children', 0) > 0


def healthy_baseline(before: dict) -> list[checks.Finding]:
    """The switch judges health as "answers like before", so before must be
    good: every unit running, each probed one listening and answering healthy,
    supervise with children."""
    return [checks.Finding(f'{u} baseline', _baseline_ok(u, b),
                           f"pid {b['pid']}, {b['status'] or 'no probe'}"
                           + (f", {b.get('children', 0)} children" if u == 'gmail-search-supervise.service' else ''))
            for u, b in before.items()]


@contextmanager
def watchdog_held(m: Machine):
    """The serve watchdog timer stopped for the block, so it cannot restart
    serve in the middle, and started again after unless it was plainly off
    (a state that cannot be read counts as running)."""
    running = m.dry_run or m.unit(WATCHDOG_TIMER, 'ActiveState') != 'inactive'
    m.act(['systemctl', '--user', 'stop', WATCHDOG_TIMER])
    try:
        yield
    finally:
        if running:
            m.act(['systemctl', '--user', 'start', WATCHDOG_TIMER])


def restart_each(m: Machine, before: dict) -> None:
    """Backend first, each waited on until it answers as it did before."""
    for unit in OWNER_UNITS:
        m.act(['systemctl', '--user', 'restart', unit])
        if not m.dry_run:
            say(wait_like_before(m, unit, before[unit]))
