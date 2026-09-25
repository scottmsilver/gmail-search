# Moving the worker VM's disk out of the dev worktree (#27)

The production worker VM (`qemu-system-x86_64 -name gmail-production-worker`) boots from
`~/development/gmail-search/worktrees/full-agents-20260915/data/production-worker/`, which is an untracked directory
inside an old dev worktree. Removing that worktree would destroy the worker. Nothing manages the VM either. The running
qemu was started by hand on 2026-09-17 and sits in a terminal's cgroup, not in a unit, so it would not come back after
a host reboot.

This runbook moves the directory to **`~/.local/share/gmail-search/worker-vm/`**, next to `public-releases/`. From
then on the VM runs from the user unit `gmail-production-worker.service`, which is enabled, and lingering is on, so it
starts at boot. The owner picks the window: deep runs are unavailable while the VM is down, normally a few minutes.

## What refers to the old path (surveyed 2026-09-25, read-only)

| Where | What |
| --- | --- |
| running qemu, pid 2787469 (`/proc/<pid>/cmdline`) | `-drive file=…/production-worker/worker.qcow2`, `…/seed.iso`, `-serial file:…/serial.log` |
| `…/production-worker/boot.sh:4` | `worker_dir=` hard-codes the directory; lines 5, 10, 11 and 14 use it |
| `gmail-search-worker-provision.service` (transient, failed 2026-09-15) | its `ExecStart` is the old `boot.sh`; cleared with `reset-failed` below |
| `~/.config/systemd/user/*` | nothing |
| `~/.config/gmail-search/*.json` | nothing. The controller and deployer reach the worker only at `127.0.0.1:22093` (`invited-runtime.json:9` `worker` block, read by `src/gmail_search/deploy/config.py:91`) |
| `src/gmail_search/deploy/`, `scripts/` | nothing |

`worker.qcow2` names its backing file relatively (`base.qcow2`), so the directory can be copied as a whole. It uses
6.8 GB on the same ext4 volume as `~/.local/share`.

## The one setting, and the only hand edit

The repo reads the new home from `worker.vm_dir` in `~/.config/gmail-search/deploy.json`. Only the move tool reads the
key. A config without it still deploys. Before the window, apply this by hand:

```diff
--- ~/.config/gmail-search/deploy.json
+++ ~/.config/gmail-search/deploy.json
@@ -24,4 +24,5 @@
     "opt_dir": "/opt/gmail-worker",
-    "image_path": "/var/lib/gmail-worker/images/agent-full.squashfs"
+    "image_path": "/var/lib/gmail-worker/images/agent-full.squashfs",
+    "vm_dir": "~/.local/share/gmail-search/worker-vm"
   }
 }
```

No other file in `~/.config/gmail-search/` changes. `invited-runtime.json` keeps `127.0.0.1:22093`, and the worker's
SSH identity and host key stay the same because the disk is the same. The tool writes the unit to
`~/.config/systemd/user/gmail-production-worker.service` itself.

## Before the window

```sh
cd ~/development/gmail-search-main          # a checkout of main that has this change
scripts/move-worker-disk.sh status
scripts/move-worker-disk.sh --dry-run move
```

The wrapper runs the checkout's `.venv/bin/python` directly, with no `uv run` and no bytecode writes. Under
`strace`, the dry run opened nothing for writing: the registry and `worker.qcow2` were opened read-only. The only
commands it ran were `git rev-parse` and `qemu-img info -U`. It refuses when:

- `worker.vm_dir` is missing
- a run is active
- the target exists but this tool did not make it
- the source and target contain each other
- the source is not self-contained: it has a symlink, or `worker.qcow2` backs onto a file outside the directory
- the source, its `boot.sh` or the target is a symlink, is not yours, or is group- or world-writable. The qemu process
  must also be yours and run `/usr/bin/qemu-system-x86_64`.
- there is not enough free space (allocated size plus 10%)
- the running qemu's command line differs from the one the tool renders, apart from the directory

It then prints the steps, the new `boot.sh` and the unit. Read them.

## The window

1. Check that no deploy or landing is running. The tool takes the landing lock
   (`~/development/gmail-search/.runtime/issue-loop/land.lock`) and refuses if the lock is held.
2. `scripts/move-worker-disk.sh move`. It:
   1. re-checks that no run is active, under the lock;
   2. powers the guest off (`sudo -n systemctl poweroff` over the admin key), then waits up to 180 s for that exact
      qemu pid to exit. If qemu does not exit, the tool stops without killing anything, and you decide what to do;
   3. copies with `cp -a --sparse=always` into a new `worker-vm.partial-XXXX` directory and checks every entry
      against the source: type, mode, size, sparseness (allocation no more than 1% over the source's) and sha256.
      Only then does it write `.moved-from`, with a fingerprint of the source tree, and rename the copy to
      `worker-vm`. A copy that fails the check is removed. The tool never deletes a directory it did not just create;
      if a crash leaves an older `worker-vm.partial-*` behind, delete it by hand;
   4. writes `worker-vm/boot.sh` and the unit, then runs `daemon-reload` and `enable`;
   5. starts the unit and waits up to 300 s for the unit to be active and SSH to answer with the pinned host key. It
      then starts the guest services the way the deployer does (`sudo -n systemctl restart gmail-full-agent-manager.service
      gmail-worker-clock-sync.service`; the manager is disabled in the guest, so nothing starts it at boot, #64) and
      waits for the manager to be active. The fallback below does the same.

   If anything after the stop fails, the tool stops the unit and boots the untouched source again as the transient
   unit `gmail-production-worker-legacy`. It then says whether the source came back. It never starts a second VM
   while one is still running. If the guest does not power off within 180 s, the tool stops before copying anything
   and kills nothing.

   The active-run check is a point-in-time check: a run started in the seconds between the check and the poweroff
   fails. Move at a quiet time.
3. Check the rest of the path by hand:
   - `systemctl --user is-active gmail-worker-gateway-tunnel.service`. The tunnel reconnects by itself within seconds.
   - Do one deep run from the browser.
4. `systemctl --user reset-failed gmail-search-worker-provision.service` clears the stale transient unit.

Re-running `move` is safe at any point:

- If the unit is already running the VM from `worker-vm`, it does nothing.
- If the tool stopped after the copy was made and nothing in the source has changed since (every entry's mode, size
  and mtime), it checks the kept copy against the source again (sha256 included) before booting it. A copy that no
  longer matches is never booted: the source comes back up, and you decide.
- If the source has run again since the copy (for example, the fallback booted it), it makes a fresh copy. The older copy
  in `worker-vm/` is renamed aside only after the fresh one verifies, then removed, and the fresh copy is the one that
  boots.

## Rollback (inside the window only)

```sh
scripts/move-worker-disk.sh --dry-run rollback
scripts/move-worker-disk.sh rollback
```

This stops and disables the unit, which powers the guest off, and boots the source recorded in
`worker-vm/.moved-from` as `gmail-production-worker-legacy`. `worker-vm/` is kept. Rollback returns the VM to the
**old disk**, so anything written to the new disk since the move is lost. That includes a deploy's worker files. Do not
roll back after a deploy has touched the worker. Fix forward instead.

## Afterwards

- The old directory is never modified or deleted by the tool. Once the worker has run well from `worker-vm/` for
  a while (a week, say), delete it by hand. After that, `worktrees/full-agents-20260915` can be removed:
  `rm -rf ~/development/gmail-search/worktrees/full-agents-20260915/data/production-worker`.
- The copy includes `keys/` and the `*-authorized-keys` files. The directory stays mode 0700.
  `~/.local/share/gmail-search` is 0775, but its parent `~/.local/share` is 0700.
- **After a host reboot**, the unit brings the VM back, but the guest manager stays down until something starts it
  (it is disabled in the guest). Until #65 is resolved, run `ssh … sudo -n systemctl restart
  gmail-full-agent-manager.service gmail-worker-clock-sync.service`, or deploy.
- Stopping the unit powers the guest off over SSH. If that fails, systemd sends qemu SIGTERM after 180 s.
- A new machine (#25) needs the `worker-vm/` directory, the `vm_dir` line above, and
  `scripts/move-worker-disk.sh` (or the unit it renders).
- `worker-vm/boot.sh` and the unit are generated. To change them, change `src/gmail_search/deploy/worker_disk.py`.
