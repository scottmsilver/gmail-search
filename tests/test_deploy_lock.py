"""The deployer's landing lock follows .claude/issue-loop/lib.sh's protocol."""
import os
import socket
import subprocess
import sys

import pytest

from gmail_search.deploy import lock


def dead_pid():
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def hold(path, line):
    path.mkdir(parents=True)
    (path / 'owner').write_text(line + '\n')


def test_acquire_writes_the_lib_sh_owner_line_and_releases(tmp_path):
    path = tmp_path / 'land.lock'
    mine = lock.acquire(path, 'deploy x')
    assert mine.startswith(f'deploy x pid={os.getpid()} host={socket.gethostname()} ')
    assert (path / 'owner').read_text().strip() == mine
    lock.release(path, mine)
    assert not path.exists()


def test_a_live_holder_is_refused(tmp_path):
    path = tmp_path / 'land.lock'
    live = f'land #5 pid={os.getpid()} host={socket.gethostname()} 2026-09-25T00:00:00Z'
    hold(path, live)
    with pytest.raises(lock.LockHeld, match='land #5'):
        lock.acquire(path, 'deploy x', wait_seconds=0)
    assert (path / 'owner').read_text().strip() == live


def test_a_live_holder_is_waited_for(tmp_path):
    path = tmp_path / 'land.lock'
    live = f'land #5 pid={os.getpid()} host={socket.gethostname()} t'
    hold(path, live)
    slept = []

    def release_on_first_sleep(seconds):
        slept.append(seconds)
        (path / 'owner').unlink()
        path.rmdir()

    mine = lock.acquire(path, 'deploy x', wait_seconds=30, poll=10, sleep=release_on_first_sleep)
    assert slept == [10] and (path / 'owner').read_text().strip() == mine


def test_a_dead_holder_on_this_host_is_taken_over(tmp_path):
    path = tmp_path / 'land.lock'
    hold(path, f'land #5 pid={dead_pid()} host={socket.gethostname()} t')
    mine = lock.acquire(path, 'deploy x')
    assert (path / 'owner').read_text().strip() == mine
    assert not list(tmp_path.glob('land.lock.stale-*'))


@pytest.mark.parametrize('line', ['land #5 host=h t', f'land #5 pid={os.getpid() + 999999} host=elsewhere t'])
def test_no_pid_or_another_host_counts_as_live(tmp_path, line):
    assert lock.holder_is_stale(line) is False


def test_release_leaves_a_lock_that_is_no_longer_ours(tmp_path):
    path = tmp_path / 'land.lock'
    mine = lock.acquire(path, 'deploy x')
    (path / 'owner').write_text('someone else pid=1 host=h t\n')
    lock.release(path, mine)
    assert path.exists()


def test_shell_and_python_share_the_mutex_file(tmp_path):
    path = tmp_path / 'land.lock'
    mine = lock.acquire(path, 'deploy x')
    assert (tmp_path / 'land.lock.mutex').exists()
    lock.release(path, mine)
