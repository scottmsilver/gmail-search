#!/usr/bin/env bash
# Run only inside the disposable Debian worker as root. Synthetic data only.
set -euo pipefail
[[ $(hostname) == synthetic-execution-worker && $(id -u) == 0 ]]
exec 9>/run/gmail-microvm-smoke.lock
flock -n 9
cd /home/worker/microvm
sha256sum --check SHA256SUMS
./firecracker --version | head -1
./jailer --version
install -m 755 firecracker jailer /usr/local/bin/
mkdir -p /srv/jailer/firecracker
jail_parent=$(mktemp -d /srv/jailer/firecracker/synthetic-hello-XXXXXXXX)
jail_id=$(basename "$jail_parent")
jail="$jail_parent/root"
mkdir -p "$jail"
install -m 444 vmlinux "$jail/vmlinux"
install -m 444 rootfs.squashfs "$jail/rootfs.squashfs"
cat > "$jail/config.json" <<'JSON'
{
  "boot-source": {
    "kernel_image_path": "/vmlinux",
    "boot_args": "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rootfstype=squashfs ro init=/bin/sh"
  },
  "drives": [{"drive_id":"rootfs","path_on_host":"/rootfs.squashfs","is_root_device":true,"is_read_only":true}],
  "machine-config": {"vcpu_count":1,"mem_size_mib":256},
  "vsock": {"guest_cid":3,"uds_path":"/gateway.vsock"}
}
JSON
chown -R 65534:65534 "$jail"
python3 - "$jail_id" <<'PY'
import ctypes, os, pathlib, signal, subprocess, sys, time
# Jailer forks when creating its PID namespace. Adopt that child so the smoke
# supervisor can wait for and terminate the actual Firecracker process.
if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
args = ['/usr/local/bin/jailer', '--id', sys.argv[1],
        '--exec-file', '/usr/local/bin/firecracker', '--uid', '65534', '--gid', '65534',
        '--new-pid-ns', '--resource-limit', 'no-file=128',
        '--resource-limit', 'fsize=16777216', '--', '--no-api', '--config-file', '/config.json']
log = pathlib.Path('/home/worker/microvm/serial.log')
with log.open('wb') as out:
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT)
    child = None
    try:
        assert proc.wait(timeout=5) == 0, 'jailer failed'
        child = int((pathlib.Path('/srv/jailer/firecracker') / sys.argv[1] / 'root/firecracker.pid').read_text())
        time.sleep(4)
        proc.stdin.write(b'mount -t proc proc /proc; mount -t sysfs sysfs /sys; echo GMAIL_SYNTHETIC_MICROVM_HELLO; /sbin/reboot -f\n')
        proc.stdin.flush()
        deadline = time.monotonic() + 30
        while True:
            pid, status = os.waitpid(child, os.WNOHANG)
            if pid:
                child = None
                result = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('synthetic microVM did not stop')
            time.sleep(.1)
    except BaseException:
        if child is not None:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
text = log.read_text(errors='replace')
print(text[-2000:])
# Match a whole line so an echoed command cannot satisfy the check.
if result != 0 or 'GMAIL_SYNTHETIC_MICROVM_HELLO' not in text.splitlines():
    raise SystemExit(f'FAIL: synthetic serial marker/clean exit missing (status={result})')
print('PASS: jailed nested Firecracker boot, guest shell marker, clean shutdown')
PY
