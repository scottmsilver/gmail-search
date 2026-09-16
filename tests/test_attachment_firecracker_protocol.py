"""A broken/untrusted peer must never escape the parser VM teardown path."""
import importlib.util
from pathlib import Path
import struct
import sys

import pytest


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'deploy/public/worker'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failure', ['oversized', 'timeout', 'disconnect'])
def test_bad_peer_always_stops_whole_vm(tmp_path, monkeypatch, failure):
    load('firecracker_backend')
    module=load('attachment_backend')
    stopped=[]
    class Backend:
        def launch(self,handle,*args):
            (tmp_path/handle/'root').mkdir(parents=True)
        def stop(self,handle):
            stopped.append(handle)
    class Peer:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def settimeout(self,*args): pass
        def sendall(self,*args): pass
        def recv(self,*args):
            if failure=='timeout': raise TimeoutError()
            return struct.pack('!I',8*1024**2+1) if failure=='oversized' else b''
    class Listener:
        def bind(self,*args): pass
        def listen(self,*args): pass
        def settimeout(self,*args): pass
        def accept(self): return Peer(),None
        def close(self): pass
    monkeypatch.setattr(module,'JAILS',tmp_path)
    monkeypatch.setattr(module.socket,'socket',lambda *args: Listener())
    monkeypatch.setattr(module.os,'chown',lambda *args: None)
    monkeypatch.setattr(module.os,'chmod',lambda *args: None)
    with pytest.raises((ValueError,TimeoutError)):
        module.AttachmentFirecrackerBackend()._run(Backend(),b'synthetic',b'{}')
    assert len(stopped)==1
