"""Fixed runner contract; synthetic subprocesses never call a provider."""
import asyncio
import importlib
import io
import json
from pathlib import Path
import struct
import sys

import pytest

WORKER=Path(__file__).parents[1]/'deploy/public/worker'
sys.path.insert(0,str(WORKER))


def envelope(profile='mail-agent-pi-v1'):
    return {'version':1,'profile':profile,'prompt':'Find invoices; $(touch /tmp/no)\nΔ',
            'tool_config':{'version':3,'tool_profile':'mail-raw-mcp-v3','capabilities':{
                'sql':'1'*64,'retrieval':'2'*64,'artifact':'3'*64,'attachment':'4'*64}},
            'inference_capability':'5'*64,'events_capability':'6'*64}


def module(name):
    assert (WORKER/(name+'.py')).exists(), 'Fixed guest runner is not implemented'
    return importlib.import_module(name)


def wire(value):
    raw=json.dumps(value).encode()
    return struct.pack('!I',len(raw))+raw


def test_prompt_roundtrip_and_historical_bootstrap_unchanged():
    boot=module('guest_agent_bootstrap')
    value=envelope()
    assert boot.read_config(io.BytesIO(wire(value)).read)==value
    old=importlib.import_module('guest_run_bootstrap')
    with pytest.raises(ValueError):old.read_config(io.BytesIO(wire(value)).read)


@pytest.mark.parametrize('mutate',[
    lambda v:v.update(owner_id='bob'),lambda v:v.update(profile='legacy-mail-v1'),
    lambda v:v.update(version=True),lambda v:v.update(prompt=''),
    lambda v:v.update(prompt='é'*8193),lambda v:v.update(prompt='a\x00b'),
    lambda v:v.update(inference_capability='provider-secret'),
    lambda v:v['tool_config'].update(version=2,tool_profile='mail-read-v2'),
    lambda v:v['tool_config']['capabilities'].update(events='7'*64),
])
def test_closed_envelope(mutate):
    boot=module('guest_agent_bootstrap'); value=envelope(); mutate(value)
    with pytest.raises(ValueError):boot.read_config(io.BytesIO(wire(value)).read)


def test_frame_bounds_duplicates_and_eof():
    boot=module('guest_agent_bootstrap')
    for payload in (struct.pack('!I',32769),wire(envelope())+b'x',b'\0\0\0\x07{"a":1}',
                    struct.pack('!I',13)+b'{"a":1,"a":2}'):
        with pytest.raises(ValueError):boot.read_config(io.BytesIO(payload).read)


@pytest.mark.asyncio
async def test_real_subprocess_receives_prompt_and_streams_before_completion(tmp_path):
    runner=module('guest_agent_pi'); marker=tmp_path/'ack'
    code='''import sys,json,time,pathlib
p=json.loads(sys.stdin.readline())
print(json.dumps({'type':'tool_execution_start','toolName':'bash','args':{'command':p['message']}}),flush=True)
while not pathlib.Path(sys.argv[1]).exists():time.sleep(.01)
print(json.dumps({'type':'message_end','message':{'role':'assistant','content':[{'type':'text','text':'answer'}],'stopReason':'stop'}}),flush=True)
print(json.dumps({'type':'agent_end'}),flush=True)
for line in sys.stdin: pass
'''
    events=[]
    async def emit(event):
        events.append(event)
        if event['type']=='tool_start':
            assert event['args']['command']==envelope()['prompt']
            marker.touch()
    proc=await asyncio.create_subprocess_exec(sys.executable,'-u','-c',code,str(marker),
        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,start_new_session=True,limit=65536)
    await runner.drive(proc,envelope()['prompt'],emit,deadline=asyncio.get_running_loop().time()+3)
    assert proc.returncode is not None
    assert [e['type'] for e in events]==['tool_start','text']
    assert events[-1]['text']=='answer'


@pytest.mark.asyncio
async def test_cancellation_reaps_child_before_return():
    runner=module('guest_agent_pi')
    proc=await asyncio.create_subprocess_exec(sys.executable,'-c','import time;time.sleep(30)',
        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,start_new_session=True)
    task=asyncio.create_task(runner.drive(proc,'hello',lambda e:None,deadline=asyncio.get_running_loop().time()+10))
    await asyncio.sleep(.03); task.cancel(); task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert proc.returncode is not None


def test_fixed_argv_and_capability_redaction():
    runner=module('guest_agent_pi')
    argv=runner.pi_argv()
    assert '--mode' in argv and argv[argv.index('--mode')+1]=='rpc'
    assert envelope()['prompt'] not in argv
    names=set(argv[argv.index('--tools')+1].split(','))
    assert len(names)==15
    assert {'bash','read','edit','write','grep','find','ls'}<=names
    event=runner.normalize({'type':'tool_execution_start','toolName':'bash','args':{'x':'5'*64}},['5'*64])
    assert '5'*64 not in json.dumps(event)


@pytest.mark.asyncio
@pytest.mark.parametrize('record',[
    '',json.dumps({'type':'agent_end'}),'{bad json}\n',
    json.dumps({'type':'message_end','message':{'role':'assistant','stopReason':'error'}}),
    json.dumps({'type':'response','success':False}),
    'x'*65537,
])
async def test_invalid_or_answerless_stream_reaps(record):
    runner=module('guest_agent_pi')
    code='import sys;sys.stdin.readline();sys.stdout.write(sys.argv[1]+"\\n");sys.stdout.flush()'
    proc=await asyncio.create_subprocess_exec(sys.executable,'-c',code,record,
        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,start_new_session=True,limit=65536)
    async def emit(event):pass
    with pytest.raises((ValueError,asyncio.IncompleteReadError)):
        await runner.drive(proc,'hello',emit,deadline=asyncio.get_running_loop().time()+2)
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_repeated_cancel_waits_for_owned_cleanup(monkeypatch):
    runner=module('guest_agent_pi'); entered=asyncio.Event(); gate=asyncio.Event()
    original=runner.stop_process
    async def slow_stop(proc):
        entered.set();await gate.wait();await original(proc)
    monkeypatch.setattr(runner,'stop_process',slow_stop)
    proc=await asyncio.create_subprocess_exec(sys.executable,'-c','import time;time.sleep(30)',
        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,start_new_session=True)
    task=asyncio.create_task(runner.drive(proc,'hello',lambda e:None,deadline=asyncio.get_running_loop().time()+5))
    await asyncio.sleep(.02);task.cancel();await entered.wait();task.cancel()
    await asyncio.sleep(.02);assert not task.done()
    gate.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_cancel_during_spawn_still_reaps_acquired_process(monkeypatch):
    runner=module('guest_agent_pi'); entered=asyncio.Event();gate=asyncio.Event();processes=[]
    original=asyncio.create_subprocess_exec
    async def gated(*args,**kwargs):
        proc=await original(*args,**kwargs);processes.append(proc);entered.set();await gate.wait();return proc
    monkeypatch.setattr(runner.asyncio,'create_subprocess_exec',gated)
    assert hasattr(runner,'start_process'), 'Spawn cancellation must retain process ownership'
    task=asyncio.create_task(runner.start_process(sys.executable,'-c','import time;time.sleep(30)',start_new_session=True))
    await entered.wait();task.cancel();await asyncio.sleep(.02);assert not task.done()
    gate.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert processes[0].returncode is not None


def test_prepare_pins_configuration_and_keeps_caps_out_of_model_file(tmp_path,monkeypatch):
    runner=module('guest_agent_pi');monkeypatch.setattr(runner,'RUN',tmp_path/'run')
    monkeypatch.setattr(runner.os,'chown',lambda *args:None)
    cwd,env=runner.prepare(envelope())
    assert cwd.stat().st_mode & 0o777 == 0o700
    persisted=json.loads((tmp_path/'run/capabilities.json').read_text())
    assert persisted==envelope()['tool_config']
    models=(tmp_path/'run/home/pi/models.json').read_text()
    assert envelope()['inference_capability'] not in models
    assert env['ANTHROPIC_API_KEY']==envelope()['inference_capability']
    assert envelope()['events_capability'] not in json.dumps(env)
    assert len(env['MCP_DIRECT_TOOLS'].split(','))==8
    with pytest.raises(FileExistsError):runner.prepare(envelope())


class FakeWriter:
    def __init__(self):self.body=b'';self.closed=False;self.aborted=False;self.transport=self
    def write(self,data):self.body+=data
    async def drain(self):pass
    def close(self):self.closed=True
    def abort(self):self.aborted=True
    async def wait_closed(self):pass


@pytest.mark.asyncio
@pytest.mark.parametrize('response',[
    b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\n\r\n{"seq":1}',
    b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\nContent-Length: 8\r\n\r\n{"seq":1}',
    b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{"seq":true}',
    b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n',
])
async def test_event_sink_rejects_bad_ack_and_closes(response,monkeypatch):
    runner=module('guest_agent_pi');reader=asyncio.StreamReader();reader.feed_data(response);reader.feed_eof();writer=FakeWriter()
    async def opened(host,port,**kwargs):
        assert (host,port)==('127.0.0.1',18080)
        return reader,writer
    monkeypatch.setattr(runner.asyncio,'open_connection',opened)
    with pytest.raises(ValueError):await runner.EventSink('6'*64)({'type':'text','text':'answer'})
    assert writer.closed and writer.aborted
    assert writer.body.startswith(b'POST /v1/events HTTP/1.1\r\n')
    assert b'Authorization: Bearer '+b'6'*64 in writer.body


@pytest.mark.asyncio
async def test_event_sink_accepts_exact_ack(monkeypatch):
    runner=module('guest_agent_pi');reader=asyncio.StreamReader()
    reader.feed_data(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 10\r\n\r\n{"seq": 1}');reader.feed_eof()
    writer=FakeWriter()
    async def opened(*args,**kwargs):return reader,writer
    monkeypatch.setattr(runner.asyncio,'open_connection',opened)
    await runner.EventSink('6'*64)({'type':'text','text':'answer'})
    assert writer.closed and writer.aborted


@pytest.mark.asyncio
async def test_full_run_uses_separate_caps_and_drains_bridge(tmp_path,monkeypatch):
    runner=module('guest_agent_pi');monkeypatch.setattr(runner,'RUN',tmp_path/'run')
    monkeypatch.setattr(runner.os,'chown',lambda *args:None)
    original=asyncio.create_subprocess_exec;processes=[];events=[]
    async def start(*args,**kwargs):
        if args[0]=='/usr/bin/python3':
            argv=[sys.executable,'-c','import time;time.sleep(30)']
            kwargs={}
        else:
            assert kwargs['env']['ANTHROPIC_API_KEY']=='5'*64
            assert '6'*64 not in json.dumps(kwargs['env'])
            argv=[sys.executable,'-u','-c',
                'import sys,json; p=json.loads(sys.stdin.readline()); '
                'print(json.dumps({"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":p["message"]}]}}),flush=True); '
                'print(json.dumps({"type":"agent_end"}),flush=True); sys.stdin.read()']
        proc=await original(*argv,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,start_new_session=True)
        processes.append(proc);return proc
    class Sink:
        def __init__(self,cap):assert cap=='6'*64
        async def __call__(self,event):events.append(event)
    monkeypatch.setattr(runner,'start_process',start);monkeypatch.setattr(runner,'EventSink',Sink)
    await runner.run(envelope())
    assert events==[{'type':'status','state':'running'}, {'type':'text','text':envelope()['prompt']},
                    {'type':'status','state':'runner_completed'}]
    assert len(processes)==2 and all(p.returncode is not None for p in processes)
    assert not (tmp_path/'run/capabilities.json').exists()


def test_model_key_uses_pi_environment_template_and_fixed_mail_guidance(tmp_path,monkeypatch):
    runner=module('guest_agent_pi');monkeypatch.setattr(runner,'RUN',tmp_path/'run')
    monkeypatch.setattr(runner.os,'chown',lambda *args:None)
    runner.prepare(envelope())
    provider=json.loads((tmp_path/'run/home/pi/models.json').read_text())['providers']['gateway']
    assert provider['apiKey']=='${ANTHROPIC_API_KEY}'
    argv=runner.pi_argv()
    assert '--append-system-prompt' in argv
    assert '[art:' in argv[argv.index('--append-system-prompt')+1]


def test_large_tool_display_is_explicitly_shortened_and_redacted_without_mutating_result():
    runner=module('guest_agent_pi')
    secret='5'*64
    result={'content':[{'type':'text','text':secret+'é'*100000}]}
    event=runner.normalize({'type':'tool_execution_end','toolName':'mail_get_thread_batch','result':result},[secret])
    assert event['result']['display_truncated'] is True
    assert event['result']['original_bytes'] > 65536
    assert secret not in json.dumps(event)
    assert len(json.dumps(event,ensure_ascii=False,separators=(',',':')).encode())<=65536
    assert result['content'][0]['text']==secret+'é'*100000


@pytest.mark.asyncio
async def test_large_rpc_tool_and_repeated_agent_end_transcript_do_not_abort_answer():
    runner=module('guest_agent_pi')
    code='''import sys,json
sys.stdin.readline()
result={'content':[{'type':'text','text':'x'*(5*1024*1024)}]}
print(json.dumps({'type':'tool_execution_end','toolName':'mail_get_thread_batch','result':result}),flush=True)
print(json.dumps({'type':'message_end','message':{'role':'assistant','content':[{'type':'text','text':'Complete answer'}]}}),flush=True)
print(json.dumps({'type':'agent_end','messages':[result,result]}),flush=True)
'''
    proc=await asyncio.create_subprocess_exec(sys.executable,'-u','-c',code,
        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,start_new_session=True,limit=32*1024**2)
    events=[]
    async def emit(event):events.append(event)
    await runner.drive(proc,'hello',emit,deadline=asyncio.get_running_loop().time()+5)
    assert [event['type'] for event in events]==['tool_result','text']
    assert events[0]['result']['display_truncated'] is True
    assert events[-1]['text']=='Complete answer'
    assert proc.returncode is not None


def test_the_gemini_profile_points_pi_at_the_gateways_google_route(tmp_path,monkeypatch):
    """Pi's Google client appends /models/<id>:streamGenerateContent to baseUrl;
    the worker relay admits exactly /v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse."""
    runner=module('guest_agent_pi');monkeypatch.setattr(runner,'RUN',tmp_path/'run')
    monkeypatch.setattr(runner.os,'chown',lambda *args:None)
    gemini=envelope('mail-agent-pi-gemini-v1')
    cwd,env=runner.prepare(gemini)
    provider=json.loads((tmp_path/'run/home/pi/models.json').read_text())['providers']['gateway']
    assert provider['api']=='google-generative-ai'
    assert provider['baseUrl']=='http://127.0.0.1:18080/v1beta'
    assert provider['apiKey']=='${GEMINI_API_KEY}'
    assert [m['id'] for m in provider['models']]==['gemini-3.8-flash']
    assert env['GEMINI_API_KEY']==gemini['inference_capability']
    assert 'ANTHROPIC_API_KEY' not in env and 'ANTHROPIC_BASE_URL' not in env
    argv=runner.pi_argv('mail-agent-pi-gemini-v1')
    assert argv[argv.index('--model')+1]=='gemini-3.8-flash'
    assert argv[argv.index('--thinking')+1]=='medium'


def test_a_non_pi_profile_is_refused_by_the_pi_runner(tmp_path,monkeypatch):
    runner=module('guest_agent_pi');monkeypatch.setattr(runner,'RUN',tmp_path/'run')
    with pytest.raises(runner.RunnerError):runner.prepare(envelope('mail-agent-claude-v1'))
