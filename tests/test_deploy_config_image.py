"""Deployer config errors name the missing key; image pin and inputs."""
import json
from pathlib import Path

import pytest

from gmail_search.deploy import image
from gmail_search.deploy.config import DeployConfigError, load_config

REPO = Path(__file__).resolve().parents[1]


def write_config(tmp_path, drop=None):
    example = json.loads((REPO / 'deploy/deploy.example.json').read_text())
    invited = tmp_path / 'invited.json'
    invited.write_text(json.dumps({'state_dir': str(tmp_path), 'worker': {'host': '127.0.0.1', 'port': 1}}))
    env = tmp_path / 'web.env'
    env.write_text('GMS_PUBLIC_ORIGIN=https://gms.example\nSECRET=x\n')
    example.update(invited_config=str(invited), public_web_env=str(env))
    if drop:
        example.pop(drop)
    path = tmp_path / 'deploy.json'
    path.write_text(json.dumps(example))
    return path


def test_example_config_loads(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.public_host == 'gms.example' and config.worker.port == 1
    assert config.registry == tmp_path / 'registry.sqlite'


def test_missing_key_names_it_and_the_example(tmp_path):
    with pytest.raises(DeployConfigError, match=r"missing key 'health_ports'.*deploy.example.json"):
        load_config(write_config(tmp_path, drop='health_ports'))


def test_missing_file_points_at_the_example(tmp_path):
    with pytest.raises(DeployConfigError, match='deploy.example.json'):
        load_config(tmp_path / 'nope.json')


def test_pin_is_rewritten_everywhere_it_is_recorded(tmp_path):
    for rel in image.PIN_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text((REPO / rel).read_text())
    old, new = image.committed_pin(tmp_path), 'a' * 64
    image.write_pin(tmp_path, new)
    assert image.committed_pin(tmp_path) == new
    assert all(old not in (tmp_path / rel).read_text() for rel in image.PIN_FILES)
    with pytest.raises(image.ImageError, match='pin mismatch'):
        image.require_pin(tmp_path, 'b' * 64)


def test_inputs_that_do_not_match_the_manifest_are_refused(tmp_path):
    for name in ('bin', 'lib', 'pi-pkgs/node_modules'):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / 'pi-pkgs/package.json').write_text('{}')
    (tmp_path / 'pi-pkgs/package-lock.json').write_text('{}')
    with pytest.raises(image.ImageError, match='do not match'):
        image.verify_inputs(REPO, tmp_path)


def test_building_without_seeded_inputs_says_how_to_seed(tmp_path):
    with pytest.raises(image.ImageError, match='--seed-image-inputs'):
        image.stage_root(REPO, tmp_path / 'cache', tmp_path / 'root')
