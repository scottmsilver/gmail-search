#!/usr/bin/env python3
"""Pi typed MCP adapter qualification; one synthetic owner's fixed bootstrap."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
sys.path.insert(0,'/tmp/runtime')
from guest_run_bootstrap import receive

ROOT=Path('/tmp/runtime')
RUN=Path('/tmp/gms-run')
MODEL='claude-sonnet-4-6'
PROMPT='SYNTHETIC_PI_MCP_TEST: Use the typed mail tools to inspect schema, query messages, fetch shared-thread, and publish pi-mcp.csv. Use bash only to create the CSV from retrieved data. Then report SYNTHETIC_PI_MCP_COMPLETE.'


def unprivileged():
    os.setgroups([]);os.setgid(1000);os.setuid(1000)
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    resource.setrlimit(resource.RLIMIT_NPROC,(128,128))


def mcp_pids():
    pids=set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            command=(path/'cmdline').read_bytes().split(b'\0')
            if b'/tmp/runtime/guest_mail_mcp.py' in command:pids.add(int(path.name))
        except (FileNotFoundError,ProcessLookupError):pass
    return pids


def main():
    config=receive()
    if config['runtime']!='pi':raise ValueError('Pi qualification only')
    RUN.mkdir(mode=0o700)
    home,cwd=RUN/'home',RUN/'work'
    home.mkdir(mode=0o700);cwd.mkdir(mode=0o700)
    for path in (RUN,home,cwd):os.chown(path,1000,1000)
    capfile=RUN/'capabilities.json'
    fd=os.open(capfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(config['capabilities'],stream)
    os.chown(capfile,1000,1000)
    del config
    (home/'pi').mkdir(mode=0o700)
    (home/'pi/models.json').write_text(json.dumps({'providers':{'synthetic':{
        'baseUrl':'http://127.0.0.1:18080','api':'anthropic-messages','apiKey':'synthetic-not-a-real-key',
        'models':[{'id':MODEL,'reasoning':False,'input':['text'],'contextWindow':200000,'maxTokens':1024,
            'compat':{'supportsEagerToolInputStreaming':False,'supportsCacheControlOnTools':False}}]}}}))
    for path in (home/'pi',home/'pi/models.json'):os.chown(path,1000,1000)
    bridge=subprocess.Popen(['/usr/bin/python3',str(ROOT/'guest-vsock-bridge.py')],stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,preexec_fn=unprivileged)
    time.sleep(.3)
    env={'HOME':str(home),'PATH':'/tmp/runtime/bin:/usr/bin:/bin','LANG':'C.UTF-8','TERM':'dumb',
         'ANTHROPIC_API_KEY':'synthetic-not-a-real-key','ANTHROPIC_BASE_URL':'http://127.0.0.1:18080',
         'DISABLE_PROMPT_CACHING':'1','PI_CODING_AGENT_DIR':str(home/'pi'),
         'MCP_DIRECT_TOOLS':'mail/describe_schema,mail/sql_query_batch,mail/get_thread_batch,mail/publish_artifact_batch'}
    argv=['/tmp/runtime/bin/node','/tmp/runtime/lib/pi-coding-agent/dist/bundle/cli.js','--provider','synthetic',
          '--model',MODEL,'--thinking','off','--tools','bash,mail_describe_schema,mail_sql_query_batch,mail_get_thread_batch,mail_publish_artifact_batch','--no-session','--no-extensions',
          '--extension','/tmp/runtime/guest-pi-mail-mcp.ts','--no-skills','--no-context-files','--no-themes',
          '--no-prompt-templates','-p',PROMPT]
    observed=set();peak=[0];done=threading.Event();monitor=None;proc=None
    def watch():
        while not done.wait(.02):
            current=mcp_pids();observed.update(current);peak[0]=max(peak[0],len(current))
    try:
        proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,preexec_fn=unprivileged)
        monitor=threading.Thread(target=watch,daemon=True);monitor.start()
        stdout,stderr=proc.communicate(timeout=95)
        done.set();monitor.join(1)
        status={'runtime':'pi','returncode':proc.returncode,'completion':b'SYNTHETIC_PI_MCP_COMPLETE' in stdout,
                'csv_exists':(cwd/'pi-mcp.csv').is_file(),'observed_mcp_processes':len(observed),'peak_mcp_processes':peak[0]}
        print('INNER_PI_MCP_RESULT '+json.dumps(status),flush=True)
        assert status['returncode']==0 and status['completion'] and status['csv_exists']
        assert status['observed_mcp_processes']==1 and status['peak_mcp_processes']==1
        print('INNER_PI_MCP_PASS',flush=True)
    finally:
        done.set()
        if monitor is not None:monitor.join(1)
        if proc is not None and proc.poll() is None:proc.kill();proc.wait(timeout=5)
        capfile.unlink(missing_ok=True)
        bridge.terminate();bridge.wait(timeout=5)


if __name__=='__main__':
    try:main()
    except Exception:
        print('INNER_PI_MCP_FAILED',flush=True)
        raise SystemExit(1)
