"""Bounded private full-agent Unix RPC server; no network listener."""
import os
from pathlib import Path
import threading
import time
import full_agent_rpc as rpc
MAX_CONNECTIONS=4
READ_SECONDS=5

class Server:
    def __init__(self, manager, socket_path, *, controller_uid):
        import socket
        self.manager = manager
        self.controller_uid = controller_uid
        self.socket_path = Path(socket_path)
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self.threads = set()
        self.threads_lock = threading.Lock()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.listener.bind(str(socket_path))
            os.chmod(socket_path, 0o660)
            self.socket_inode = self.socket_path.lstat().st_ino
            self.listener.listen(MAX_CONNECTIONS)
            self.listener.settimeout(.1)
        except BaseException:
            self.listener.close()
            raise

    def _connection(self, peer):
        import socket
        import struct
        try:
            with peer:
                _, uid, _ = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != self.controller_uid:
                    return
                end = time.monotonic() + READ_SECONDS
                def read(size):
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('RPC input deadline')
                    peer.settimeout(remaining)
                    return peer.recv(size)
                request, payload = rpc.read_frame(read)
                if read(1) != b'':
                    raise rpc.ProtocolError('Trailing RPC input')
                response, output = self.manager.handle(request, payload, peer_uid=uid)
                del payload
                peer.settimeout(READ_SECONDS)
                peer.sendall(rpc.encode_frame(response, output, response=True))
        except Exception:
            # Invalid/incomplete requests get EOF, never a success acknowledgement.
            pass
        finally:
            with self.threads_lock:
                self.threads.discard(threading.current_thread())
            self.slots.release()

    def serve(self, stop_event):
        import socket
        while not stop_event.is_set():
            self.manager.tick()
            try:
                peer, _ = self.listener.accept()
            except socket.timeout:
                continue
            if not self.slots.acquire(blocking=False):
                peer.close()
                continue
            thread = threading.Thread(target=self._connection, args=(peer,), daemon=False)
            with self.threads_lock:
                self.threads.add(thread)
            try:
                thread.start()
            except BaseException:
                peer.close()
                with self.threads_lock:
                    self.threads.discard(thread)
                self.slots.release()
                raise

    def close(self):
        self.listener.close()
        with self.threads_lock:
            threads = tuple(self.threads)
        for thread in threads:
            thread.join()
        try:
            if self.socket_path.lstat().st_ino == self.socket_inode:
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

