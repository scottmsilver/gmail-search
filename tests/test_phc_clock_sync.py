"""The worker clock daemon: steps to the host clock only when it has drifted."""
import importlib.util
from pathlib import Path
import time

import pytest

SOURCE = Path(__file__).parents[1] / 'deploy/public/worker/production/phc_clock_sync.py'


@pytest.fixture
def sync():
    spec = importlib.util.spec_from_file_location('phc_clock_sync', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_clocks(monkeypatch, sync, host_ahead):
    wall = {'now': 1000.0}
    host = object()
    stepped = []
    def gettime(clock):
        return 1000.0 + host_ahead if clock is host else wall['now']
    def settime(clock, value):
        assert clock == time.CLOCK_REALTIME
        stepped.append(value - wall['now'])
        wall['now'] = value
    monkeypatch.setattr(sync.time, 'clock_gettime', gettime)
    monkeypatch.setattr(sync.time, 'clock_settime', settime)
    return host, stepped


def test_a_drifted_clock_is_stepped_to_the_host(monkeypatch, sync):
    """The drift that refused every launch: worker 5.4 s behind the controller."""
    host, stepped = fake_clocks(monkeypatch, sync, host_ahead=5.4)
    sync.step_if_drifted(host)
    assert stepped == [pytest.approx(5.4)]
    assert sync.offset_from_host(host) == pytest.approx(0)


def test_a_clock_within_threshold_is_left_alone(monkeypatch, sync):
    host, stepped = fake_clocks(monkeypatch, sync, host_ahead=0.01)
    sync.step_if_drifted(host)
    assert stepped == []


def test_the_dynamic_clock_id_matches_the_kernel_macro(sync):
    # FD_TO_CLOCKID(fd) = ((~(clockid_t)(fd)) << 3) | CLOCKFD, CLOCKFD = 3
    assert sync.phc_clock_id(3) == -29
