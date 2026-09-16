"""One-shot synthetic run credentials cannot select paths or commands."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import pytest


def module():
    path=Path(__file__).parents[1]/'deploy/public/worker/guest_run_bootstrap.py'
    spec=importlib.util.spec_from_file_location('guest_run_bootstrap',path)
    loaded=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:spec.loader.exec_module(loaded)
    finally:sys.path.pop(0)
    return loaded


def config():
    return dict(version=1,runtime='pi',capabilities={'sql':'1'*64,'retrieval':'2'*64,'artifact':'3'*64})


def wire(value):
    raw=json.dumps(value).encode()
    return struct.pack('!I',len(raw))+raw


def test_bootstrap_valid_and_one_shot_eof():
    mod=module()
    assert mod.read_config(io.BytesIO(wire(config())).read)==config()
    with pytest.raises(ValueError):
        mod.read_config(io.BytesIO(wire(config())+b'trailing').read)


@pytest.mark.parametrize('mutate', [lambda c:c.update(owner_id='bob'), lambda c:c.update(command='sh'),
    lambda c:c.update(runtime='other'), lambda c:c.update(version=True),
    lambda c:c['capabilities'].update(sql='invalid'),lambda c:c['capabilities'].update(provider='4'*64)])
def test_bootstrap_closed_schema(mutate):
    mod=module();value=config();mutate(value)
    with pytest.raises(ValueError):mod.read_config(io.BytesIO(wire(value)).read)


def test_bootstrap_size_and_duplicate_reject_before_input():
    mod=module()
    stream=io.BytesIO(struct.pack('!I',4097)+b'not-read')
    with pytest.raises(ValueError):mod.read_config(stream.read)
    assert stream.tell()==4
    raw=b'{"version":1,"version":1}'
    with pytest.raises(ValueError):mod.read_config(io.BytesIO(struct.pack('!I',len(raw))+raw).read)


def test_explicit_v2_bootstrap_persistence_roundtrip(tmp_path):
    mod=module()
    value={**config(),'version':2,'tool_profile':'mail-read-v2',
           'capabilities':{**config()['capabilities'],'attachment':'4'*64}}
    tmp_path.chmod(0o700)
    with pytest.raises(ValueError):mod.persist_config(tmp_path,value)
    assert not (tmp_path/'capabilities.json').exists()
    parsed=mod.read_config(io.BytesIO(wire(value)).read,expected_profile='mail-read-v2')
    mod.persist_config(tmp_path,parsed,expected_profile='mail-read-v2')
    persisted=json.loads((tmp_path/'capabilities.json').read_bytes())
    assert persisted=={key:value[key] for key in ('version','tool_profile','capabilities')}
    with pytest.raises(ValueError):mod.read_config(io.BytesIO(wire(value)+b'extra').read,expected_profile='mail-read-v2')
