#!/usr/bin/env python3
"""Unprivileged forced-SSH frontend: one bounded request to one fixed socket.

Installed root-owned alongside attachment_rpc.py. No arguments, shell commands,
user-supplied destinations, sudo, or interpretation of SSH_ORIGINAL_COMMAND.
"""
import os
import select
import socket
import sys
import time

if __name__ == '__main__':
    sys.path.insert(0, '/opt/gmail-worker')
import attachment_rpc as rpc

SOCKET_PATH = '/run/gmail-attachment-rpc/manager.sock'
READ_SECONDS = 5
RPC_SECONDS = 30


def relay(input_fd, output_fd, *, socket_path=SOCKET_PATH):
    end = time.monotonic() + READ_SECONDS
    def read_input(size):
        remaining = end - time.monotonic()
        if remaining <= 0 or not select.select([input_fd], [], [], remaining)[0]:
            raise TimeoutError('RPC input deadline')
        return os.read(input_fd, size)
    request, payload = rpc.read_frame(read_input)
    if read_input(1) != b'':
        raise rpc.ProtocolError('Trailing RPC input')
    encoded = rpc.encode_frame(request, payload)
    # Deadline includes root teardown and transport; root independently bounds
    # reads and the VMM independently enforces its short and hard leases.
    end = time.monotonic() + RPC_SECONDS
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(RPC_SECONDS)
        peer.connect(socket_path)
        peer.sendall(encoded)
        peer.shutdown(socket.SHUT_WR)
        del encoded, payload
        def read_peer(size):
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('RPC response deadline')
            peer.settimeout(remaining)
            return peer.recv(size)
        response, payload = rpc.read_frame(read_peer, response=True)
        if read_peer(1) != b'':
            raise rpc.ProtocolError('Trailing RPC response')
    if any(response[k] != request[k] for k in rpc.COMMON - {'payload_size'}):
        raise rpc.ProtocolError('RPC response binding mismatch')
    encoded = memoryview(rpc.encode_frame(response, payload, response=True))
    end = time.monotonic() + READ_SECONDS
    os.set_blocking(output_fd, False)
    while encoded:
        remaining = end - time.monotonic()
        if remaining <= 0 or not select.select([], [output_fd], [], remaining)[1]:
            raise TimeoutError('RPC output deadline')
        try:
            count = os.write(output_fd, encoded[:65536])
        except BlockingIOError:
            continue
        if count <= 0:
            raise OSError('RPC output closed')
        encoded = encoded[count:]


def main():
    if len(sys.argv) != 1:
        return 1
    try:
        relay(sys.stdin.fileno(), sys.stdout.fileno())
    except Exception:
        # No parser errors, payloads, request IDs or user context in SSH stderr.
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
