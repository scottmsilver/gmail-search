"""Keep the worker's wall clock on the hypervisor's.

The worker has no route to an NTP server, so systemd-timesyncd never syncs and
the clock drifts. Lease deadlines are compared across the controller and this
VM with a 5 s margin; at 5.4 s of drift every launch was refused as `expired`.
The KVM PTP device (`ptp_kvm`, /dev/ptp0) reads the host's CLOCK_REALTIME, so
stepping to it needs no network and no extra package.
"""
import os
import sys
import time

PHC = '/dev/ptp0'
INTERVAL_SECONDS = 30
STEP_THRESHOLD_SECONDS = 0.05


def phc_clock_id(fd):
    """The dynamic POSIX clock id for an open PTP device (FD_TO_CLOCKID)."""
    return ((~fd) << 3) | 3


def offset_from_host(clock):
    before = time.clock_gettime(time.CLOCK_REALTIME)
    host = time.clock_gettime(clock)
    after = time.clock_gettime(time.CLOCK_REALTIME)
    return host - (before + after) / 2


def step_if_drifted(clock):
    offset = offset_from_host(clock)
    if abs(offset) > STEP_THRESHOLD_SECONDS:
        time.clock_settime(time.CLOCK_REALTIME, time.clock_gettime(time.CLOCK_REALTIME) + offset)
        print(f'stepped wall clock by {offset:+.3f}s', flush=True)


def main():
    fd = os.open(PHC, os.O_RDONLY)
    clock = phc_clock_id(fd)
    while True:
        step_if_drifted(clock)
        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    sys.exit(main())
