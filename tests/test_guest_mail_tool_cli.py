"""Bound raw guest invocation input before parsing; configuration is never a tool argument."""
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]/'deploy/public/worker'


def module():
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location('guest_mail_tool_cli', ROOT/'guest_mail_tool_cli.py')
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    finally:
        sys.path.pop(0)


@pytest.mark.asyncio
async def test_cli_reads_private_fixed_config_and_passes_no_identity_arguments(tmp_path):
    cli = module()
    cli.RUN_ROOT = tmp_path/'run'
    cli.RUN_ROOT.mkdir(mode=0o700)
    config = cli.RUN_ROOT/'capabilities.json'
    config.write_text(json.dumps({'sql':'a'*64,'retrieval':'b'*64,'artifact':'c'*64}))
    config.chmod(0o600)
    seen = []
    class Tools:
        def __init__(self, workspace, capabilities, *, port):
            seen.append((workspace, capabilities, port))
        async def dispatch(self, name, args):
            seen.append((name,args))
            return {'synthetic':42}
        async def aclose(self):
            seen.append('closed')
    cli.GuestMailTools = Tools
    result = await cli.invoke(io.BytesIO(b'{"name":"describe_schema","arguments":{}}'))
    assert result == {'synthetic':42}
    assert seen[0][0] == cli.RUN_ROOT/'work' and seen[0][2] == 18080
    assert seen[0][1].profile == 'legacy-mail-v1'
    assert dict(seen[0][1].capabilities) == {'sql':'a'*64,'retrieval':'b'*64,'artifact':'c'*64}
    assert seen[1] == ('describe_schema',{})
    assert seen[-1]=='closed'


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', [b'x'*(256*1024+1), b'{', b'{"name":"x","name":"y","arguments":{}}', b'{"name":"x","arguments":{},"port":80}'], ids=['oversize', 'malformed', 'duplicate', 'routing'])
async def test_cli_rejects_invalid_or_oversized_raw_input_before_configuration(raw):
    cli = module()
    with pytest.raises(cli.ToolError):
        await cli.invoke(io.BytesIO(raw))


@pytest.mark.asyncio
async def test_cli_rejects_symlink_or_public_credentials(tmp_path):
    cli = module()
    cli.RUN_ROOT = tmp_path/'run'
    cli.RUN_ROOT.mkdir(mode=0o700)
    config = cli.RUN_ROOT/'capabilities.json'
    config.write_text('{}')
    config.chmod(0o644)
    request = b'{"name":"describe_schema","arguments":{}}'
    with pytest.raises(cli.ToolError):
        await cli.invoke(io.BytesIO(request))
    config.unlink()
    target = tmp_path/'target'
    target.write_text('{}')
    target.chmod(0o600)
    config.symlink_to(target)
    with pytest.raises(cli.ToolError):
        await cli.invoke(io.BytesIO(request))


@pytest.mark.asyncio
@pytest.mark.parametrize('result', [{'text':'\ud800'}, {'number':float('inf')}], ids=['surrogate', 'overflow'])
async def test_cli_sanitizes_unencodable_gateway_result(result, monkeypatch):
    cli = module()
    output = bytearray()
    monkeypatch.setattr(cli.os, 'set_blocking', lambda *args: None)
    def write(fd, data):
        output.extend(data)
        return len(data)
    monkeypatch.setattr(cli.os, 'write', write)
    assert await cli._output(result) is False
    assert json.loads(output) == {'error':'Invalid guest tool result.'}


def test_cli_subprocess_rejects_malformed_input_without_traceback():
    import subprocess
    result = subprocess.run([sys.executable, '-I', str(ROOT/'guest_mail_tool_cli.py')], input=b'{', capture_output=True, timeout=5)
    assert result.returncode == 1
    assert json.loads(result.stdout) == {'error':'Guest invocation failed.'}
    assert result.stderr == b''


def test_cli_subprocess_cancellation_closes_waiting_input():
    import subprocess
    import time
    process = subprocess.Popen([sys.executable, '-I', str(ROOT/'guest_mail_tool_cli.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(.2)
        process.terminate()
        # Keep stdin open until cancellation completes; communicate() closes it
        # and would race EOF against SIGTERM, exercising a different outcome.
        process.wait(timeout=3)
        output, error = process.communicate(timeout=3)
        assert process.returncode == 130
        assert 'cancelled' in json.loads(output)['error']
        assert error == b''
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_cli_preserves_private_v2_envelope_and_rejects_mixed_audiences(tmp_path):
    cli=module()
    cli.RUN_ROOT=tmp_path/'run'
    cli.RUN_ROOT.mkdir(mode=0o700)
    value={'version':2,'tool_profile':'mail-read-v2','capabilities':{'sql':'1'*64,'retrieval':'2'*64,'artifact':'3'*64,'attachment':'4'*64}}
    config=cli.RUN_ROOT/'capabilities.json'
    config.write_text(json.dumps(value));config.chmod(0o600)
    result=cli._capabilities()
    assert result.profile=='mail-read-v2' and result.capabilities['attachment']=='4'*64
    config.write_text(json.dumps(value['capabilities']))
    with pytest.raises(cli.ToolError):cli._capabilities()
