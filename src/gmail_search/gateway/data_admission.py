"""Shared, non-queued per-process capacity for trusted database operations.

Composition must supply the same instance to analytical SQL and fixed search
readers. Cross-process deployments require a separately qualified shared budget.
"""
from threading import Lock
from types import MappingProxyType


class _Lease:
    def __init__(self, admission, owner_id):
        self._admission = admission
        self._owner_id = owner_id
        self._released = False

    def release(self):
        admission = self._admission
        with admission._lock:
            if self._released:
                return
            self._released = True
            remaining = admission._active[self._owner_id] - 1
            if remaining:
                admission._active[self._owner_id] = remaining
            else:
                del admission._active[self._owner_id]


class DataAdmission:
    def __init__(self, *, global_concurrency=8, owner_concurrency=2):
        if (type(global_concurrency) is not int or type(owner_concurrency) is not int
                or not 1 <= owner_concurrency <= global_concurrency <= 32):
            raise ValueError('Invalid data query capacity')
        self._global_concurrency = global_concurrency
        self._owner_concurrency = owner_concurrency
        self._lock = Lock()
        self._active = {}

    @property
    def global_concurrency(self):
        return self._global_concurrency

    @property
    def owner_concurrency(self):
        return self._owner_concurrency

    @property
    def active(self):
        with self._lock:
            return MappingProxyType(dict(self._active))

    def acquire(self, owner_id):
        if not isinstance(owner_id,str) or not owner_id or len(owner_id)>2048 or '\x00' in owner_id:
            raise ValueError('Invalid data owner')
        with self._lock:
            if (sum(self._active.values()) >= self.global_concurrency
                    or self._active.get(owner_id,0) >= self.owner_concurrency):
                raise RuntimeError('Data query capacity reached')
            self._active[owner_id] = self._active.get(owner_id,0) + 1
            return _Lease(self,owner_id)
