"""Deploy configuration, read from machine config files; nothing host-specific
lives in code. A missing key fails naming the key and the checked-in example."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_PATH = Path('~/.config/gmail-search/deploy.json')
EXAMPLE = 'deploy/deploy.example.json'


class DeployConfigError(ValueError):
    pass


@dataclass(frozen=True)
class WorkerAccess:
    host: str
    port: int
    user: str
    key: Path
    known_hosts: Path
    opt_dir: str
    image_path: str
    # Host directory holding the worker VM's disk and boot files (#27). Only
    # the disk-move tool reads it, so a config without it still deploys.
    vm_dir: Path | None = None


@dataclass(frozen=True)
class DeployConfig:
    install_root: Path
    registry: Path
    public_web_env: Path
    public_host: str
    web_node_modules: Path
    web_local_url: str
    health_host: str
    health_ports: tuple[int, ...]
    image_inputs_cache: Path
    services: dict[str, str]
    worker: WorkerAccess

    @property
    def releases(self) -> Path:
        return self.install_root / 'public-releases'

    @property
    def current_links(self) -> tuple[Path, Path]:
        return self.install_root / 'invited-current', self.install_root / 'public-current'


def _require(data: dict, key: str, where: Path):
    if key not in data:
        raise DeployConfigError(f"{where}: missing key '{key}' (see {EXAMPLE})")
    return data[key]


def _path(value: str) -> Path:
    return Path(value).expanduser()


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; values are never logged by the deployer."""
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _public_host(env_path: Path) -> str:
    origin = read_env_file(env_path).get('GMS_PUBLIC_ORIGIN')
    if not origin:
        raise DeployConfigError(f"{env_path}: missing key 'GMS_PUBLIC_ORIGIN'")
    return urlsplit(origin).netloc


def load_config(path: Path = DEFAULT_PATH) -> DeployConfig:
    path = path.expanduser()
    if not path.exists():
        raise DeployConfigError(f'{path} does not exist; copy {EXAMPLE} there and edit it')
    data = json.loads(path.read_text())
    req = lambda key, where=data: _require(where, key, path)  # noqa: E731
    invited_path = _path(req('invited_config'))
    invited = json.loads(invited_path.read_text())
    endpoint = _require(invited, 'worker', invited_path)
    worker = req('worker')
    env_path = _path(req('public_web_env'))
    services = req('services')
    for name in ('controller', 'web', 'manager', 'clock_sync'):
        req(name, services)
    return DeployConfig(
        install_root=_path(req('install_root')),
        registry=_path(_require(invited, 'state_dir', invited_path)) / 'registry.sqlite',
        public_web_env=env_path, public_host=_public_host(env_path),
        web_node_modules=_path(req('web_node_modules')),
        web_local_url=req('web_local_url').rstrip('/'), health_host=req('health_host'),
        health_ports=tuple(int(p) for p in req('health_ports')),
        image_inputs_cache=_path(req('image_inputs_cache')), services=dict(services),
        worker=WorkerAccess(
            host=_require(endpoint, 'host', invited_path), port=int(_require(endpoint, 'port', invited_path)),
            user=req('user', worker), key=_path(req('key', worker)),
            known_hosts=_path(req('known_hosts', worker)),
            opt_dir=req('opt_dir', worker), image_path=req('image_path', worker),
            vm_dir=_path(worker['vm_dir']) if 'vm_dir' in worker else None))
