#!/usr/bin/env python3
"""Run real CLI binaries against a synthetic local provider in a clean worker.

Run as the unprivileged worker inside a fresh network namespace containing only
loopback. Inputs are public runtime binaries plus the pure request normalizer.
No production credentials, data, network access or application package required.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path('/home/worker/cli-spike')
MODEL='claude-sonnet-4-6'
PROMPT='SYNTHETIC_CLI_SPIKE: Use the shell tool to run Python, write synthetic-result.txt containing 42, and report SYNTHETIC_CLI_COMPLETE.'
COMMAND="python3 -c \"from pathlib import Path; Path('synthetic-result.txt').write_text(str(6 * 7)); print('SYNTHETIC_TOOL_OK 42')\""


def frame(kind,**fields):
    return ('event: '+kind+'\ndata: '+json.dumps(dict(type=kind,**fields))+'\n\n').encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve-runtime',choices=('pi','claude'))
    parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--ready-file',type=Path)
    args=parser.parse_args()
    if not 0<=args.port<=65535:parser.error('Invalid port')
    if socket.gethostname()!='synthetic-execution-worker' or os.geteuid()==0 or (not args.serve_runtime and sorted(name for _,name in socket.if_nameindex())!=['lo']):
        raise SystemExit('Requires unprivileged clean worker in a loopback-only namespace')
    os.umask(0o077)
    spec=importlib.util.spec_from_file_location('normalizer',ROOT/'inference.py')
    normalizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(normalizer)
    output=Path(tempfile.mkdtemp(prefix='cli-results.',dir=ROOT));output.chmod(0o700)
    sys.path.insert(0,str(ROOT/'python'))
    import httpx
    from gateway.registry import Registry
    from gateway.capabilities import Capabilities
    from gateway.provider import AnthropicRunService, ProviderProfile
    from gateway.provider_http import AnthropicHTTPTransport
    observations=[]
    active={}
    class Wire(httpx.AsyncByteStream):
        def __init__(self,payload):self.payload=payload
        async def __aiter__(self):yield self.payload
    async def through_gateway(body,payload,record):
        async def upstream(request):
            normalized=json.loads(request.content)
            record['upstream_fields']=sorted(normalized)
            record['upstream_model']=normalized['model']
            record['server_effort']=normalized.get('output_config',{}).get('effort')
            if 'metadata' in normalized or normalized.get('output_config')!={'effort':'high'}:
                raise ValueError('Compatibility hint unexpectedly forwarded')
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Wire(payload))
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream),trust_env=False) as client:
            service=AnthropicRunService(active['capabilities'],AnthropicHTTPTransport('synthetic-upstream-key',client=client))
            result=b''.join([chunk async for chunk in service.stream(active['token'],'request-'+str(len(observations)),body)])
            record['gateway_chain']='accepted'
            return result
    def configure(runtime,base):
        state=base/'registry';state.mkdir(mode=0o700)
        registry=Registry(state/'registry.sqlite',is_active=lambda owner:owner=='synthetic')
        budget=registry.create_budget('synthetic',10**9)
        run=registry.start_run('synthetic','conversation',request_key='start',budget_id=budget,lease_ttl=600,deadline_ttl=600)
        capabilities=Capabilities(registry)
        active.update(capabilities=capabilities,token=capabilities.issue(run.run_id,audience='inference',operations=['generate'],ttl=600).secret)
        profile=ProviderProfile(MODEL,200000,32768,1,1,client_profile='pi-0.84.4' if runtime=='pi' else 'claude-2.1.272',effort='high')
        AnthropicRunService(capabilities,None).bind_profile(run.run_id,profile)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=8*1024*1024:self.send_error(413);return
            body=json.loads(self.rfile.read(length))
            current=len(observations)
            (output/f'request-{current}.json').write_text(json.dumps(body,indent=2))
            record={'path':self.path,'fields':sorted(body),'model':body.get('model'),
                    'header_names':sorted(self.headers),'tools':[t.get('name',t.get('type')) for t in body.get('tools',[])]}
            try:
                compiler=normalizer.compile_anthropic_count_tokens_request if 'count_tokens' in self.path else normalizer.compile_anthropic_request
                compiler(body,server_model=MODEL,max_output_tokens=131072)
                record['strict_normalizer']='accepted'
            except ValueError:record['strict_normalizer']='rejected'
            observations.append(record)
            if 'count_tokens' in self.path:
                payload=json.dumps({'input_tokens':100}).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload);return
            if not self.path.startswith('/v1/messages'):
                self.send_error(404);return
            tool_results=[c for m in body.get('messages',[]) for c in (m.get('content',[]) if isinstance(m.get('content'),list) else []) if c.get('type')=='tool_result' and c.get('tool_use_id')=='synthetic_tool']
            model=body['model']
            tool_name=next((t.get('name') for t in body.get('tools',[]) if t.get('name','').lower()=='bash'),None)
            tool=not tool_results and tool_name is not None
            if tool_results:record['tool_result_received']=True
            message={'id':'msg_synthetic','type':'message','role':'assistant','model':model,'content':[], 'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':100,'output_tokens':1}}
            frames=[frame('message_start',message=message)]
            if tool:
                frames.extend([frame('content_block_start',index=0,content_block={'type':'tool_use','id':'synthetic_tool','name':tool_name,'input':{}}),
                    frame('content_block_delta',index=0,delta={'type':'input_json_delta','partial_json':json.dumps({'command':COMMAND,'description':'Run synthetic Python check'})})])
            else:
                frames.extend([frame('content_block_start',index=0,content_block={'type':'text','text':''}),
                    frame('content_block_delta',index=0,delta={'type':'text_delta','text':'SYNTHETIC_CLI_COMPLETE'})])
            frames.extend([frame('content_block_stop',index=0),frame('message_delta',delta={'stop_reason':'tool_use' if tool else 'end_turn','stop_sequence':None},usage={'output_tokens':20}),frame('message_stop')])
            payload=b''.join(frames)
            try:
                payload=asyncio.run(through_gateway(body,payload,record))
            except Exception:
                record['gateway_chain']='rejected'
                self.send_error(400,'Synthetic gateway rejected request');return
            (output/'observations.json').write_text(json.dumps(observations,indent=2))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload);self.wfile.flush()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    endpoint=f'http://127.0.0.1:{server.server_port}'
    summary=[]
    try:
        if args.serve_runtime:
            base=output/args.serve_runtime;base.mkdir()
            configure(args.serve_runtime,base)
            status={'endpoint':endpoint,'runtime':args.serve_runtime,'server_effort':'high','output':str(output),'pid':os.getpid()}
            if args.ready_file:args.ready_file.write_text(json.dumps(status))
            print(json.dumps(status),flush=True)
            stopping=threading.Event()
            signal.signal(signal.SIGTERM,lambda *_:stopping.set())
            signal.signal(signal.SIGINT,lambda *_:stopping.set())
            stopping.wait(600)
            (output/'observations.json').write_text(json.dumps(observations,indent=2))
            return
        for runtime in ('pi','claude'):
            base=output/runtime;base.mkdir();home=base/'home';home.mkdir();cwd=base/'work';cwd.mkdir()
            env={'HOME':str(home),'PATH':str(ROOT/'bin')+':/usr/bin:/bin','LANG':'C.UTF-8','TERM':'dumb',
                 'ANTHROPIC_API_KEY':'synthetic-not-a-real-key','ANTHROPIC_BASE_URL':endpoint,
                 'DISABLE_PROMPT_CACHING':'1','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
                 'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS':'1','CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK':'1',
                 'CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL':'1','CLAUDE_CODE_DISABLE_AUTO_MEMORY':'1',
                 'CLAUDE_CODE_DISABLE_THINKING':'1','CLAUDE_CONFIG_DIR':str(home/'claude'),'PI_CODING_AGENT_DIR':str(home/'pi')}
            if runtime=='pi':
                (home/'pi').mkdir()
                (home/'pi/models.json').write_text(json.dumps({'providers':{'synthetic':{'baseUrl':endpoint,'api':'anthropic-messages','apiKey':'synthetic-not-a-real-key','models':[{'id':MODEL,'reasoning':False,'input':['text'],'contextWindow':200000,'maxTokens':1024,'compat':{'supportsEagerToolInputStreaming':False,'supportsCacheControlOnTools':False}}]}}}))
                command=[str(ROOT/'bin/node'),str(ROOT/'lib/pi-coding-agent/dist/bundle/cli.js'),'--provider','synthetic','--model',MODEL,'--thinking','off','--tools','bash','--no-session','--no-extensions','--no-skills','--no-context-files','--no-themes','--no-prompt-templates','-p',PROMPT]
            else:
                command=[str(ROOT/'bin/claude'),'--bare','--setting-sources','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}','--disable-slash-commands','--no-session-persistence','--model',MODEL,'--tools','Bash','--allowedTools','Bash','--dangerously-skip-permissions','--max-turns','3','--output-format','json','--system-prompt','You are a synthetic local tool test.','-p',PROMPT]
            configure(runtime,base)
            start=len(observations)
            try:
                result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,timeout=60)
                (base/'stdout').write_bytes(result.stdout);(base/'stderr').write_bytes(result.stderr)
                summary.append({'runtime':runtime,'returncode':result.returncode,'requests':len(observations)-start,
                    'tool_file':(cwd/'synthetic-result.txt').read_text() if (cwd/'synthetic-result.txt').exists() else None,
                    'completion_marker':b'SYNTHETIC_CLI_COMPLETE' in result.stdout})
            except subprocess.TimeoutExpired:
                summary.append({'runtime':runtime,'timeout':True,'requests':len(observations)-start})
        (output/'observations.json').write_text(json.dumps(observations,indent=2))
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({'output':str(output),'summary':summary,'observations':observations},indent=2))
    finally:
        server.shutdown();server.server_close()
        if args.ready_file:args.ready_file.unlink(missing_ok=True)

if __name__=='__main__':main()
