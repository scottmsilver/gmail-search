#!/usr/bin/env python3
"""Immutable, synthetic-only CLI exercise with one-shot run capabilities."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0,'/tmp/runtime')
from guest_run_bootstrap import receive

ROOT=Path('/tmp/runtime')
RUN=Path('/tmp/gms-run')
MODEL='claude-sonnet-4-6'
PROMPT='SYNTHETIC_MAIL_TOOL_TEST: Use Bash to run /usr/bin/python3 -I /tmp/runtime/guest-mail-workflow.py and then report SYNTHETIC_MAIL_TOOLS_COMPLETE.'


def unprivileged():
    os.setgroups([]);os.setgid(1000);os.setuid(1000)
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    resource.setrlimit(resource.RLIMIT_NPROC,(128,128))


def main():
    config=receive()
    runtime=config['runtime']
    RUN.mkdir(mode=0o700)
    home,cwd=RUN/'home',RUN/'work'
    home.mkdir(mode=0o700);cwd.mkdir(mode=0o700)
    for path in (RUN,home,cwd):os.chown(path,1000,1000)
    capfile=RUN/'capabilities.json'
    fd=os.open(capfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(config['capabilities'],stream)
    os.chown(capfile,1000,1000)
    del config
    bridge=subprocess.Popen(['/usr/bin/python3',str(ROOT/'guest-vsock-bridge.py')],stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,preexec_fn=unprivileged)
    time.sleep(.3)
    env={'HOME':str(home),'PATH':'/tmp/runtime/bin:/usr/bin:/bin','LANG':'C.UTF-8','TERM':'dumb',
         'ANTHROPIC_API_KEY':'synthetic-not-a-real-key','ANTHROPIC_BASE_URL':'http://127.0.0.1:18080',
         'GMS_SYNTHETIC_RUNTIME':runtime,'DISABLE_PROMPT_CACHING':'1','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
         'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS':'1','CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK':'1',
         'CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL':'1','CLAUDE_CODE_DISABLE_AUTO_MEMORY':'1',
         'CLAUDE_CODE_DISABLE_THINKING':'1','CLAUDE_CONFIG_DIR':str(home/'claude'),'PI_CODING_AGENT_DIR':str(home/'pi')}
    if runtime=='pi':
        (home/'pi').mkdir()
        (home/'pi/models.json').write_text(json.dumps({'providers':{'synthetic':{
            'baseUrl':'http://127.0.0.1:18080','api':'anthropic-messages','apiKey':'synthetic-not-a-real-key',
            'models':[{'id':MODEL,'reasoning':False,'input':['text'],'contextWindow':200000,'maxTokens':1024,
                'compat':{'supportsEagerToolInputStreaming':False,'supportsCacheControlOnTools':False}}]}}}))
        for path in (home/'pi',home/'pi/models.json'):os.chown(path,1000,1000)
        argv=['/tmp/runtime/bin/node','/tmp/runtime/lib/pi-coding-agent/dist/bundle/cli.js','--provider','synthetic',
              '--model',MODEL,'--thinking','off','--tools','bash','--no-session','--no-extensions','--no-skills',
              '--no-context-files','--no-themes','--no-prompt-templates','-p',PROMPT]
    else:
        argv=['/tmp/runtime/bin/claude','--bare','--setting-sources','','--strict-mcp-config','--mcp-config',
              '{"mcpServers":{}}','--disable-slash-commands','--no-session-persistence','--model',MODEL,
              '--tools','Bash','--allowedTools','Bash','--dangerously-skip-permissions','--max-turns','3',
              '--output-format','json','--system-prompt','You are a synthetic local tool test.','-p',PROMPT]
    try:
        proc=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,preexec_fn=unprivileged,timeout=95)
        assert proc.returncode==0 and b'SYNTHETIC_MAIL_TOOLS_COMPLETE' in proc.stdout
        assert (cwd/(runtime+'-mail.csv')).is_file()
        print('INNER_MAIL_TOOLS_RESULT '+json.dumps({'runtime':runtime,'returncode':proc.returncode,
            'completion':True,'csv_bytes':(cwd/(runtime+'-mail.csv')).stat().st_size}),flush=True)
        print('INNER_MAIL_TOOLS_PASS',flush=True)
    finally:
        capfile.unlink(missing_ok=True)
        bridge.terminate();bridge.wait(timeout=5)


if __name__=='__main__':
    try:main()
    except Exception:
        print('INNER_MAIL_TOOLS_FAILED',flush=True)
        raise SystemExit(1)
