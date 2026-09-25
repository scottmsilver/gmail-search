"""Process execution. Always an argument list, never a shell, so a release
name, ref or path cannot become shell syntax."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


class CommandFailed(RuntimeError):
    def __init__(self, args, code, log=None):
        where = f', see {log}' if log else ''
        super().__init__(f"{' '.join(map(str, args))} exited {code}{where}")
        self.code = code
        self.log = log


class Runner:
    """The real runner. Tests substitute a fake with the same two methods."""

    def run(self, args, *, cwd=None, env=None, log: Path | None = None, check=True) -> int:
        full_env = {**os.environ, **env} if env else None
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
        full_env = {**os.environ, **env} if env else None
        done = subprocess.run(list(map(str, args)), cwd=cwd, env=full_env, capture_output=True, text=True)
        if check and done.returncode != 0:
            raise CommandFailed(args, done.returncode)
        return done.stdout
