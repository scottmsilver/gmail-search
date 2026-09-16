#!/usr/bin/env python3
"""Full-agent guest-local byte bridge; fixed gateway route and 12 MiB ceiling.

No destination supplied by client; outer relay still enforces HTTP routes,
run ownership and request quotas. No mailbox or provider credential is present.
"""
import select
import socket
import threading
import time

LIMIT = 12 * 1024**2
SLOTS = threading.BoundedSemaphore(8)


def bridge(client):
    try:
        with client, socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM) as upstream:
            upstream.settimeout(5)
            upstream.connect((2, 8000))
            peers = {client: upstream, upstream: client}
            sizes = {client: 0, upstream: 0}
            deadline = time.monotonic() + 90
            while peers and time.monotonic() < deadline:
                ready, _, _ = select.select(list(peers), [], [], .25)
                for source in ready:
                    chunk = source.recv(65536)
                    if not chunk:
                        return
                    sizes[source] += len(chunk)
                    if sizes[source] > LIMIT:
                        return
                    peers[source].sendall(chunk)
    except (OSError, TimeoutError):
        pass
    finally:
        SLOTS.release()


def serve():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 18080))
        server.listen(8)
        while True:
            client, _ = server.accept()
            if not SLOTS.acquire(blocking=False):
                client.close()
                continue
            client.settimeout(5)
            threading.Thread(target=bridge, args=(client,), daemon=True).start()


if __name__ == '__main__':
    serve()
