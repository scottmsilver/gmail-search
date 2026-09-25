"""Process execution. Always an argument list, never a shell, so a release
name, ref or path cannot become shell syntax."""
from __future__ import annotations

import functools
import os
from pathlib import Path
import subprocess


@functools.cache
def _repository_env_vars() -> frozenset[str]:
    """The variables that pick a repository, index or config for git."""
    names = subprocess.run(['git', 'rev-parse', '--local-env-vars'], capture_output=True, text=True).stdout.split()
    return frozenset(names) | {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY'}


def without_git_env(env=None) -> dict[str, str]:
    """The environment minus the variables that pick git's repository. A git
    hook exports GIT_DIR, GIT_INDEX_FILE and friends; inherited, they aim every
    `git -C <path>` at the hook's repository instead of <path>. Transport
    settings (GIT_SSH_COMMAND) stay, so `git fetch` still reaches origin."""
    drop = _repository_env_vars()
    return {k: v for k, v in (os.environ if env is None else env).items()
            if k not in drop and not k.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_'))}


class CommandFailed(RuntimeError):
    def __init__(self, args, code, log=None):
        where = f', see {log}' if log else ''
        super().__init__(f"{' '.join(map(str, args))} exited {code}{where}")
        self.code = code
        self.log = log


class Runner:
    """The real runner. Tests substitute a fake with the same two methods."""

    def run(self, args, *, cwd=None, env=None, log: Path | None = None, check=True) -> int:
        full_env = {**without_git_env(), **(env or {})}
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, 'wb') as sink:
                code = subprocess.run(list(map(str, args)), cwd=cwd, env=full_env, stdout=sink,
                                      stderr=subprocess.STDOUT).returncode
        else:
            code = subprocess.run(list(map(str, args)), cwd=cwd, env=full_env).returncode
        if check and code != 0:
            raise CommandFailed(args, code, log)
        return code

    def capture(self, args, *, cwd=None, env=None, check=True) -> str:
        full_env = {**without_git_env(), **(env or {})}
        done = subprocess.run(list(map(str, args)), cwd=cwd, env=full_env, capture_output=True, text=True)
        if check and done.returncode != 0:
            raise CommandFailed(args, done.returncode)
        return done.stdout
